import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.foundad import VisionModule


def _grid_size(num_tokens: int) -> int | None:
    side = int(round(num_tokens ** 0.5))
    return side if side * side == num_tokens else None


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # Adapted from microsoft/Swin-Transformer window_partition.
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    # Adapted from microsoft/Swin-Transformer window_reverse.
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class SupportAggregator(nn.Module):
    """Aggregate K support images into one normal support token map."""

    def __init__(self, embed_dim: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.weight = nn.Linear(embed_dim, 1)
        elif mode != "mean":
            raise ValueError(f"Unsupported support aggregation: {mode}")

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        # support: [B, K, N, D]
        if self.mode == "mean":
            return support.mean(dim=1)
        logits = self.weight(support).squeeze(-1)  # [B, K, N]
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * support).sum(dim=1)


class SupportQueryMatcher(nn.Module):
    """Top-m semantic matcher with a soft relative-position prior."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        top_m: int = 8,
        n_heads: int = 8,
        use_rel_pos: bool = True,
        rel_pos_weight: float = 0.1,
        max_grid: int = 32,
        matcher_mode: str = "window",
        window_size: int = 4,
        window_shift: int = 0,
    ):
        super().__init__()
        self.top_m = top_m
        self.n_heads = n_heads
        self.use_rel_pos = use_rel_pos
        self.rel_pos_weight = rel_pos_weight
        self.matcher_mode = matcher_mode
        self.window_size = window_size
        self.window_shift = window_shift
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        if matcher_mode not in {"global", "window"}:
            raise ValueError(f"Unsupported matcher_mode: {matcher_mode}")
        if use_rel_pos:
            self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, 2 * max_grid - 1, 2 * max_grid - 1))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self._rel_cache = {}

    def _relative_bias(self, n: int, device: torch.device) -> torch.Tensor | None:
        grid = _grid_size(n)
        if grid is None or not self.use_rel_pos:
            return None
        key = (grid, device)
        if key not in self._rel_cache:
            coords = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), dim=-1).view(-1, 2)
            rel = coords[:, None, :] - coords[None, :, :]
            center = self.rel_pos_bias.shape[1] // 2
            rel = rel + center
            rel = rel.clamp(0, self.rel_pos_bias.shape[1] - 1).to(device)
            self._rel_cache[key] = rel
        rel = self._rel_cache[key]
        return self.rel_pos_bias[:, rel[..., 0], rel[..., 1]].mean(dim=0)

    def _topm_context(self, sim: torch.Tensor, values: torch.Tensor):
        m = min(self.top_m, sim.shape[-1])
        top_sim, top_idx = torch.topk(sim, k=m, dim=-1)
        top_weights = torch.softmax(top_sim, dim=-1)
        expanded = values.unsqueeze(1).expand(values.shape[0], sim.shape[1], values.shape[1], values.shape[2])
        gather_idx = top_idx.unsqueeze(-1).expand(values.shape[0], sim.shape[1], m, values.shape[-1])
        top_values = torch.gather(expanded, dim=2, index=gather_idx)
        context = torch.einsum("bnm,bnmd->bnd", top_weights, top_values)
        return context, top_weights, top_sim

    def _forward_windowed(self, z_q: torch.Tensor, z_s: torch.Tensor, q: torch.Tensor, k: torch.Tensor, grid: int):
        bsz, _, dim = z_q.shape
        window_size = min(self.window_size, grid)
        if window_size <= 0 or grid % window_size != 0:
            return None

        q_grid = q.view(bsz, grid, grid, q.shape[-1])
        k_grid = k.view(bsz, grid, grid, k.shape[-1])
        z_s_grid = z_s.view(bsz, grid, grid, dim)

        shift = self.window_shift % window_size if self.window_shift else 0
        if shift:
            q_grid = torch.roll(q_grid, shifts=(-shift, -shift), dims=(1, 2))
            k_grid = torch.roll(k_grid, shifts=(-shift, -shift), dims=(1, 2))
            z_s_grid = torch.roll(z_s_grid, shifts=(-shift, -shift), dims=(1, 2))

        q_win = _window_partition(q_grid, window_size).view(-1, window_size * window_size, q.shape[-1])
        k_win = _window_partition(k_grid, window_size).view(-1, window_size * window_size, k.shape[-1])
        z_s_win = _window_partition(z_s_grid, window_size).view(-1, window_size * window_size, dim)

        sim = torch.bmm(q_win, k_win.transpose(1, 2)) / math.sqrt(q_win.shape[-1])
        bias = self._relative_bias(window_size * window_size, z_q.device)
        if bias is not None:
            sim = sim + self.rel_pos_weight * bias.unsqueeze(0)

        context_win, top_weights_win, top_sim_win = self._topm_context(sim, z_s_win)
        context_grid = _window_reverse(context_win.view(-1, window_size, window_size, dim), window_size, grid, grid)
        weight_grid = _window_reverse(top_weights_win.view(-1, window_size, window_size, top_weights_win.shape[-1]), window_size, grid, grid)
        sim_grid = _window_reverse(top_sim_win.view(-1, window_size, window_size, top_sim_win.shape[-1]), window_size, grid, grid)

        if shift:
            context_grid = torch.roll(context_grid, shifts=(shift, shift), dims=(1, 2))
            weight_grid = torch.roll(weight_grid, shifts=(shift, shift), dims=(1, 2))
            sim_grid = torch.roll(sim_grid, shifts=(shift, shift), dims=(1, 2))

        return (
            context_grid.view(bsz, grid * grid, dim),
            weight_grid.view(bsz, grid * grid, -1),
            sim_grid.view(bsz, grid * grid, -1),
        )

    def forward(self, z_q: torch.Tensor, z_s: torch.Tensor):
        # z_q, z_s: [B, N, D]
        q = F.normalize(self.q_proj(z_q), dim=-1)
        k = F.normalize(self.k_proj(z_s), dim=-1)
        grid = _grid_size(z_q.shape[1])
        if self.matcher_mode == "window" and grid is not None:
            windowed = self._forward_windowed(z_q, z_s, q, k, grid)
            if windowed is not None:
                return windowed

        sim = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        bias = self._relative_bias(z_q.shape[1], z_q.device)
        if bias is not None:
            sim = sim + self.rel_pos_weight * bias.unsqueeze(0)

        return self._topm_context(sim, z_s)


class ConditionalPredictor(nn.Module):
    """Predict expected normal query embeddings conditioned on support context."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(3 * embed_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, z_q: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_q, context, z_q - context], dim=-1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return z_q + self.output_proj(x)


class RelevanceGate(nn.Module):
    """Estimate which query tokens are relevant to the support-defined object."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_q: torch.Tensor, context: torch.Tensor, top_weights: torch.Tensor):
        max_weight = top_weights.max(dim=-1, keepdim=True).values
        entropy = -(top_weights * (top_weights + 1e-8).log()).sum(dim=-1, keepdim=True)
        x = torch.cat([z_q, context, max_weight, entropy], dim=-1)
        return self.gate_mlp(x).squeeze(-1)


class SupportConditionedPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 384,
        depth: int = 4,
        top_m: int = 8,
        n_heads: int = 8,
        aggregation: str = "mean",
        use_rel_pos: bool = True,
        rel_pos_weight: float = 0.1,
        dropout: float = 0.0,
        matcher_mode: str = "window",
        window_size: int = 4,
        window_shift: int = 0,
    ):
        super().__init__()
        self.aggregator = SupportAggregator(embed_dim, mode=aggregation)
        self.matcher = SupportQueryMatcher(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            top_m=top_m,
            n_heads=n_heads,
            use_rel_pos=use_rel_pos,
            rel_pos_weight=rel_pos_weight,
            matcher_mode=matcher_mode,
            window_size=window_size,
            window_shift=window_shift,
        )
        self.predictor = ConditionalPredictor(embed_dim, hidden_dim, n_heads=n_heads, n_layers=depth, dropout=dropout)
        self.gate = RelevanceGate(embed_dim)

    def forward(self, query: torch.Tensor, support: torch.Tensor):
        # query: [B, N, D], support: [B, K, N, D]
        support_agg = self.aggregator(support)
        context, top_weights, top_sim = self.matcher(query, support_agg)
        pred = self.predictor(query, context)
        relevance = self.gate(query, context, top_weights)
        residual = (query - pred).pow(2).sum(dim=-1)
        score = relevance * residual
        return {
            "pred": pred,
            "support_agg": support_agg,
            "context": context,
            "top_weights": top_weights,
            "top_sim": top_sim,
            "gate": relevance,
            "residual": residual,
            "score": score,
        }


class SupportConditionedVisionModule(VisionModule):
    def __init__(
        self,
        model_name: str,
        pred_depth: int,
        pred_emb_dim: int,
        use_cuda: bool = True,
        if_pe: bool = True,
        feat_normed: bool = False,
        encoder_cfg=None,
        top_m: int = 8,
        n_heads: int = 8,
        aggregation: str = "mean",
        use_rel_pos: bool = True,
        rel_pos_weight: float = 0.1,
        dropout: float = 0.0,
        matcher_mode: str = "window",
        window_size: int = 4,
        window_shift: int = 0,
        **_: object,
    ):
        super().__init__(
            model_name=model_name,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            use_cuda=use_cuda,
            if_pe=if_pe,
            feat_normed=feat_normed,
            encoder_cfg=encoder_cfg,
        )
        self.predictor = SupportConditionedPredictor(
            embed_dim=self.embed_dim,
            hidden_dim=pred_emb_dim,
            depth=pred_depth,
            top_m=top_m,
            n_heads=n_heads,
            aggregation=aggregation,
            use_rel_pos=use_rel_pos,
            rel_pos_weight=rel_pos_weight,
            dropout=dropout,
            matcher_mode=matcher_mode,
            window_size=window_size,
            window_shift=window_shift,
        )
        if use_cuda and torch.cuda.is_available():
            self.cuda()

    def support_features(self, images: torch.Tensor, paths, n_layer: int = 3):
        # images: [B, K, C, H, W]
        b, k, c, h, w = images.shape
        flat_paths = [p for group in paths for p in group] if paths is not None else [""] * (b * k)
        feats = self.target_features(images.view(b * k, c, h, w), flat_paths, n_layer=n_layer)
        return feats.view(b, k, feats.shape[1], feats.shape[2])

    def predict(self, query: torch.Tensor, support: torch.Tensor):
        return self.predictor(query, support)
