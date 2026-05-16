# FoundAD <img src="./assets/icon.png" alt="FoundAD logo showing a stylized magnifying glass over abstract shapes, representing anomaly detection in visual data. The logo is set against a neutral background and does not contain any text. The tone is professional and focused." width="30" height="30">

The implementation of the paper **Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors** ([arXiv](http://arxiv.org/abs/2510.01934
), [OpenReview](https://openreview.net/forum?id=YRrlJ8oVEH)).

  <a href="https://ymxlzgy.com/">Guangyao Zhai</a>, <a href="https://karolinezhy.github.io/">Yue Zhou</a>, <a href="">Xinyan Deng</a>, <a href="https://scholar.google.com/citations?user=f5DnPiEAAAAJ&hl=de">Lars Heckler</a>, <a href="https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/">Nassir Navab</a>, and <a href="https://www.cs.cit.tum.de/camp/members/benjamin-busam/">Benjamin Busam</a>
<br>
  Technical University of Munich <span style="margin: 0 10px;">•</span> MVTec Software GmbH

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Training and Inference](#train-infer)
   - [Dataset Preparation](#dataset-preparation)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
4. [Acknowledgement](#acknowledgement)
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n foundad python=3.10
conda activate foundad
git clone git@github.com:ymxlzgy/FoundAD.git
cd FoundAD
pip install -r requirements.txt
pip install -e .
```


## Quick Start
Before we start, please make sure you have the rights to use [DINOv3](https://github.com/facebookresearch/dinov3). Download our trained manifold projectors, and put them to `./logs/`. 
|DINOv3-based|1-shot|2-shot|4-shot|
|---------|:---------:|:---------:|:---------:|
|**MVTec AD**|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_4shot.zip)|
|**VisA**  |[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_4shot.zip)|


Run a demo on MVTec-AD 
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=mvtec data.data_name=mvtec_1shot data.test_root=assets/mvtec
```

Or a demo on VisA
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=visa data.data_name=visa_4shot data.test_root=assets/visa
```

### DINOv3 Torch Hub cache and local weights

This fork avoids writing to the global `~/.cache/torch` by default. Torch Hub cache is placed under `.cache/torch` inside this repo unless you override it with `SCJEPA_TORCH_HOME`, `SCJEPA_TORCH_HUB_DIR`, or the matching Hydra keys under `app.meta`.

To load DINOv3 from a local checkout and local checkpoint:

```bash
python foundad/main.py mode=train app=train_scjepa_dinov3 \
  app.meta.dinov3_repo=/path/to/dinov3 \
  app.meta.dinov3_source=local \
  app.meta.dinov3_model=dinov3_vitb16 \
  app.meta.dinov3_weights=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  data.dataset=mvtec data.data_name=mvtec_4shot data.data_path=/path/to/fewshot-root
```

For ViT-L/16, set `app.meta.dinov3_model=dinov3_vitl16` and point `app.meta.dinov3_weights` to the matching `.pth`.


## Training and Inference

### Experimental: Support-Conditioned JEPA

This fork also includes a simple FoundAD-compatible variant named **Support-Conditioned JEPA**. It keeps the frozen foundation encoder, but replaces the unconditional manifold projector with a few-shot support-conditioned latent predictor:

```text
query tokens + normal support tokens -> expected normal query tokens
```

Architecture:

```text
support images -> frozen DINOv3 -> support tokens
query image    -> frozen DINOv3 -> query tokens

support aggregation      : mean or learned position-wise aggregation
support-query matcher    : top-m semantic token matching with optional relative-position bias
conditional predictor    : transformer over [z_q, C, z_q - C]
relevance gate           : MLP over [z_q, C, max_match_weight, match_entropy]
anomaly score            : gate * ||z_q - z_hat_q||^2
```

Training uses mixed multi-class episodes. Each episode samples K normal support images and one normal query from the same category. To stay close to the original FoundAD setting, query input follows the same 50/50 clean-vs-CutPaste branch: clean episodes train identity consistency, and CutPaste episodes train prediction toward the clean query feature. The gate uses a small synthetic latent-deviation auxiliary loss by default; set `meta.gate_weight=0.0` to use only `L_pred + identity_weight * L_id`.

Train it with:

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_4shot data.data_path=/path/to/fewshot-root app=train_scjepa_dinov3 diy_name=_scjepa
```

Evaluate it with:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_4shot data.test_root=/path/to/mvtec app=test_scjepa app.ckpt_step=1950 diy_name=_scjepa
```

The new code lives in `foundad/src/support_jepa.py`, `foundad/src/train_support.py`, and `foundad/src/AD_support.py`.

### Dataset Preparation

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: [<u>Here</u>](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | We use the structured dataset of [<u>RealNet</u>](https://github.com/cnulab/RealNet). |

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/media/ymxlzgy/Data21/xinyan/visa target=/media/ymxlzgy/Data21/xinyan/visa_tmp seed=42 num_samples=2
```
where `source` is the dataset folder, `target` is the folder of few-shot samples, and `num_samples` is the number of samples training models, e.g., 2 for 2-shot learning. `seed` can be adjusted to have multiple rounds of experiment.

### Model Training

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/media/ymxlzgy/Data21/xinyan app=train_dinov3 diy_name=dbug
```
where `data.dataset` is "mvtec" or "visa", `data.data_name` is the folder name of few-shot samples, `data.data_path` is the path where the few-shot folder is at, `app` is "train_dinov3" or other model configs under `configs/app/`, and `diy_name` (optionally) is the post-fix name of the model saving directory. To adjust the layer, please specify `app.meta.n_layer`.

### Anomaly Detection / Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot diy_name=dbug data.test_root=/media/ymxlzgy/Data21/xinyan/mvtec app=test app.ckpt_step=1950
```
where `data.test_root` is the dataset folder, and `app` is test_dinov2 or test_dinov3 under `configs/app/`. To adjust sample number K, please specify `testing.K_top_mvtec` and `testing.K_top_visa`.

## Acknowledgement
This repo utilizes [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2](https://github.com/facebookresearch/dinov2), [DINO](https://github.com/facebookresearch/dino), [SigLIP](https://github.com/google-research/big_vision), [CLIP](https://github.com/openai/CLIP) and [DINOSigLIP](https://github.com/tri-ml/prismatic-vlms). We also thank [I-JEPA](https://github.com/facebookresearch/ijepa) for the inspiration.
