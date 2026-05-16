from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms

from src.datasets.dataset import build_base_transform, build_dataloader
from src.helper import init_opt
from src.support_jepa import SupportConditionedVisionModule
from src.utils.logging import AverageMeter, CSVLogger, gpu_timer
from src.utils.synthesis import CutPasteUnion

random.seed(42)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportBank:
    def __init__(self, train_root: str, resize: int, shots: int):
        self.root = Path(train_root) / "train"
        self.shots = shots
        self.transform = transforms.Compose(build_base_transform(resize))
        self.feature_cache = None
        self.paths = {
            p.name: sorted([x for x in p.iterdir() if x.is_file()])
            for p in self.root.iterdir()
            if p.is_dir()
        }
        if not self.paths:
            raise FileNotFoundError(f"No class folders found under {self.root}")

    def sample(self, labels: List[str], avoid_paths: List[str] | None = None):
        images, path_groups = [], []
        avoid_paths = avoid_paths or [""] * len(labels)
        for label, avoid in zip(labels, avoid_paths):
            candidates = self.paths[label]
            pool = [p for p in candidates if str(p) != avoid] or candidates
            if len(pool) >= self.shots:
                chosen = random.sample(pool, self.shots)
            else:
                chosen = random.choices(pool, k=self.shots)
            imgs = [self.transform(Image.open(p).convert("RGB")) for p in chosen]
            images.append(torch.stack(imgs, dim=0))
            path_groups.append([str(p) for p in chosen])
        return torch.stack(images, dim=0), path_groups

    @torch.inference_mode()
    def build_feature_cache(self, model, device: torch.device, n_layer: int, use_bf16: bool = False):
        feature_cache = {}
        model.encoder.eval()
        for label, paths in self.paths.items():
            imgs = [self.transform(Image.open(p).convert("RGB")) for p in paths]
            imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            path_strs = [str(p) for p in paths]
            with autocast(dtype=torch.bfloat16, enabled=use_bf16):
                feats = model.target_features(imgs, path_strs, n_layer=n_layer)
            feature_cache[label] = {
                "paths": path_strs,
                "features": feats.detach(),
            }
            logger.info("Cached %d support features for class %s", len(path_strs), label)
        self.feature_cache = feature_cache

    def sample_features(self, labels: List[str], avoid_paths: List[str] | None = None):
        if self.feature_cache is None:
            raise RuntimeError("Support feature cache has not been built.")

        features, path_groups = [], []
        avoid_paths = avoid_paths or [""] * len(labels)
        for label, avoid in zip(labels, avoid_paths):
            cached = self.feature_cache[label]
            paths = cached["paths"]
            pool = [i for i, path in enumerate(paths) if path != avoid] or list(range(len(paths)))
            if len(pool) >= self.shots:
                chosen = random.sample(pool, self.shots)
            else:
                chosen = random.choices(pool, k=self.shots)
            idx = torch.as_tensor(chosen, device=cached["features"].device, dtype=torch.long)
            features.append(cached["features"].index_select(0, idx))
            path_groups.append([paths[i] for i in chosen])
        return torch.stack(features, dim=0), path_groups


class SupportTrainer:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        mcfg = args["meta"]
        self.model = SupportConditionedVisionModule(
            mcfg["model"],
            mcfg["pred_depth"],
            mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True),
            feat_normed=mcfg.get("feat_normed", False),
            top_m=mcfg.get("top_m", 8),
            n_heads=mcfg.get("n_heads", 8),
            aggregation=mcfg.get("support_aggregation", "mean"),
            use_rel_pos=mcfg.get("use_rel_pos", True),
            rel_pos_weight=mcfg.get("rel_pos_weight", 0.1),
            dropout=mcfg.get("dropout", 0.0),
            encoder_cfg=mcfg,
        )
        self.n_layer = mcfg.get("n_layer", 3)
        self.loss_mode = mcfg.get("loss_mode", "l2")
        self.identity_weight = float(mcfg.get("identity_weight", 0.25))
        self.gate_weight = float(mcfg.get("gate_weight", 0.05))
        self.support_shots = int(mcfg.get("support_shots", args["data"].get("support_shots", 4)))
        self.use_bf16 = mcfg["use_bfloat16"]

        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)

        dcfg = args["data"]
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            num_workers=dcfg.get("num_workers", 8),
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip", False),
            use_vflip=dcfg.get("use_vflip", False),
            use_rotate90=dcfg.get("use_rotate90", False),
            use_color_jitter=dcfg.get("use_color_jitter", False),
            use_gray=dcfg.get("use_gray", False),
            use_blur=dcfg.get("use_blur", False),
        )
        self.support_bank = SupportBank(dcfg["train_root"], mcfg["crop_size"], self.support_shots)
        self.support_bank.build_feature_cache(self.model, self.device, self.n_layer, self.use_bf16)
        self.cutpaste = CutPasteUnion(colorJitter=0.5)

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            use_bfloat16=self.use_bf16,
            max_epoch=ocfg["epochs"],
            min_lr=ocfg.get("min_lr", 1e-6),
            warmup_epoch=ocfg.get("warmup_epoch", 5),
            step_size=ocfg.get("step_size", 300),
            gamma=ocfg.get("gamma", 0.1),
        )
        self.epochs = ocfg["epochs"]

        lcfg = args.get("logging", {})
        self.ckpt_dir = Path(lcfg.get("folder", "logs"))
        self.tag = lcfg.get("write_tag", "train")
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%.5f", "recon"),
            ("%.5f", "identity"),
            ("%.5f", "gate"),
            ("%d", "gpu_time_ms"),
            ("%d", "data_time_ms"),
            ("%d", "iter_time_ms"),
        )

    def _loss_fn(self, pred, target):
        if self.loss_mode == "l2":
            return F.mse_loss(pred.flatten(0, 1), target.flatten(0, 1), reduction="mean")
        if self.loss_mode == "smooth_l1":
            return F.smooth_l1_loss(pred.flatten(0, 1), target.flatten(0, 1), reduction="mean")
        raise NotImplementedError(self.loss_mode)

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save(
            {
                "predictor": self.model.predictor.state_dict(),
                "projector": self.model.projector.state_dict() if self.model.projector else None,
                "epoch": ep,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
            self.ckpt_dir / name,
        )

    def train(self):
        mp.set_start_method("spawn", force=True)
        gstep = 0
        for ep in range(self.epochs):
            logger.info("Epoch %d", ep + 1)
            self.sampler.set_epoch(ep)
            loss_m, gpu_time_m, data_time_m, iter_time_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            loader_iter = iter(self.loader)
            itr = 0
            data_start = time.perf_counter()
            while True:
                try:
                    imgs, labels, paths = next(loader_iter)
                except StopIteration:
                    break
                data_ms = (time.perf_counter() - data_start) * 1000.0
                iter_start = time.perf_counter()
                imgs = imgs.to(self.device, non_blocking=True)
                _, imgs_abn = self.cutpaste(imgs, labels)
                use_clean = np.random.rand() < 0.5
                query_input = imgs if use_clean else imgs_abn
                support, _ = self.support_bank.sample_features(list(labels), list(paths))

                def _step():
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        target = self.model.target_features(imgs, paths, n_layer=self.n_layer)
                        query = target if use_clean else self.model.target_features(query_input, paths, n_layer=self.n_layer)
                        out = self.model.predict(query, support)
                        pred = out["pred"]
                        recon = self._loss_fn(pred, target)
                        identity = self._loss_fn(pred, query) if use_clean else pred.new_tensor(0.0)
                        gate_loss = pred.new_tensor(0.0)
                        if self.gate_weight > 0:
                            latent_delta = (query.detach() - target.detach()).pow(2).mean(dim=-1)
                            gate_target = (latent_delta > latent_delta.mean(dim=1, keepdim=True)).float()
                            gate_loss = F.binary_cross_entropy(out["gate"].clamp(1e-4, 1 - 1e-4), gate_target)
                        loss = recon + self.identity_weight * identity + self.gate_weight * gate_loss
                        return loss, recon, identity, gate_loss

                (loss, recon, identity, gate_loss), t = gpu_timer(lambda: _step())
                if self.use_bf16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                iter_ms = (time.perf_counter() - iter_start) * 1000.0
                loss_m.update(loss.item())
                gpu_time_m.update(t)
                data_time_m.update(data_ms)
                iter_time_m.update(iter_ms)
                gstep += 1
                if gstep % 100 == 0:
                    self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep + 1, itr, loss.item(), recon.item(), identity.item(), gate_loss.item(), t, data_ms, iter_ms)
                if itr % 100 == 0:
                    logger.info(
                        "[E %d I %d] loss %.6f recon %.6f id %.6f gate %.6f avg %.6f mem %.2fMB gpu %.1fms data %.1fms iter %.1fms",
                        ep + 1,
                        itr,
                        loss.item(),
                        recon.item(),
                        identity.item(),
                        gate_loss.item(),
                        loss_m.avg,
                        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                        gpu_time_m.avg,
                        data_time_m.avg,
                        iter_time_m.avg,
                    )
                itr += 1
                data_start = time.perf_counter()
            if self.scheduler is not None:
                self.scheduler.step()


def main(args: Dict[str, Any]) -> None:
    SupportTrainer(args).train()
