______________________________________________________________________

<div align="center">

# PAND: Prompt-Aware Neighborhood Distillation for Lightweight Fine-Grained Visual Classification

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://pytorchlightning.ai/">
  <img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://hydra.cc/">
  <img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>

</div>

---

## 📋 Abstract

Distilling knowledge from large Vision-Language Models (VLMs) into lightweight networks is crucial yet challenging in Fine-Grained Visual Classification (FGVC), due to the reliance on fixed prompts and global alignment. To address this, we propose PAND (Prompt-Aware Neighborhood Distillation), a two-stage framework that decouples semantic calibration from structural transfer. First, we incorporate Prompt-Aware Semantic Calibration to generate adaptive semantic anchors. Second, we introduce a neighborhood-aware structural distillation strategy to constrain the student's local decision structure. PAND consistently outperforms state-of-the-art methods on four FGVC benchmarks. Notably, our ResNet-18 student achieves 76.09% accuracy on CUB-200, surpassing the strong baseline VL2Lite by 3.4%. 

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd PAND

# 2. Create conda environment
conda create -n pand_env python=3.9
conda activate pand_env

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt
pip install open-clip-torch
```

---

## 📁 Dataset Configuration

### 1. Dataset Config Files Location
```
configs/data/attributes/
├── 0_CUB_200_2011.yaml          # CUB-200-2011 dataset
├── 1_FGVC_AIRCRAFT.yaml         # FGVC Aircraft dataset
├── 4_OxfordIIITPet.yaml         # Oxford Pets dataset
├── 5_StanfordDogs.yaml          # Stanford Dogs dataset
└── ...
```

### 2. Modify Dataset Path
Edit the config file for your dataset:

```yaml
# Example: configs/data/attributes/0_CUB_200_2011.yaml
data_root: /path/to/your/CUB_200_2011  # Change this to your dataset path
class_num: 200
prompt_tmpl: "a photo of a {}, a type of bird."
classes:
  1: "Black_footed_Albatross"
  2: "Laysan_Albatross"
  # ... (class names)
```

**Key fields to modify:**
- `data_root`: Path to your dataset directory
- `class_num`: Number of classes
- `prompt_tmpl`: Template for text prompts (optional)
- `classes`: Class ID to name mapping

---

## 🎓 Stage-PSC: Prompt Semantic Calibration

Run the automated script:

```bash
bash scripts/run_coop_stagePSC.sh
```

**What this script does:**
1. Extracts CLIP image features
2. Trains CoOp with pre-extracted features
3. Generates `learned_text_features.pt` for Stage-NSD

**Configuration:**
Edit `scripts/run_coop_stagePSC.sh` to modify:
- `DATASET`: Dataset name (e.g., `0_CUB_200_2011`)
- `DATA_ROOT`: Path to your dataset
- `GPUS`: GPU devices to use

---

## 🎯 Stage-NSD: Neighborhood Semantic Distillation

```bash
python src/train.py \
  data/attributes=0_CUB_200_2011 \
  model=coop_kd \
  model.net.student.model_name=resnet18 \
  model.net.teacher.coop_text_features=/path/to/learned_text_features.pt \
  model.kd_criterion.use_coop=true \
  model.kd_criterion.use_nlrd=true \
  model.kd_criterion.nlrd_k=3 \
  model.kd_criterion.nlrd_lambda=1.0 \
  model.kd_criterion.nlrd_weight=1.0 \
  trainer=ddp \
  trainer.devices=4 \
  trainer.max_epochs=300 \
  data.batch_size=128
```

---

## 🔧 Key Parameters

### Stage-PSC (Prompt Semantic Calibration)
- `n_ctx`: Number of learnable context tokens (default: 16)
- `lr`: Learning rate (default: 0.002)
- `epochs`: Training epochs (default: 200)
- `batch_size`: Batch size (default: 32)

### Stage-NSD (Neighborhood Semantic Distillation)
- `model`: Config file (`kda` for baseline, `coop_kd` for PAND)
- `model.net.student.model_name`: Student architecture (`resnet18`, `mobilenet_v2`)
- `model.kd_criterion.use_coop`: Enable CoOp text features
- `model.kd_criterion.use_nlrd`: Enable NLRD loss
- `model.kd_criterion.nlrd_weight`: NLRD loss weight (0~1)
- `trainer.devices`: Number of GPUs
- `data.batch_size`: Batch size (recommend 128 for 4 GPUs)

---

## 📚 References

This project builds upon the following excellent works:

- **VL2Lite**: [Visual-Language Knowledge Distillation Framework](https://github.com/jsjangAI/VL2Lite)
- **CoOp**: [Context Optimization for Prompt Learning](https://github.com/KaiyangZhou/CoOp)
- **NRKD**: [Neighborhood-based Relational Knowledge Distillation](https://github.com/xinxiaoxiaomeng/NRKD.git)

---

## 📄 Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{luo2026pandpromptawareneighborhooddistillation,
      title={PAND: Prompt-Aware Neighborhood Distillation for Lightweight Fine-Grained Visual Classification}, 
      author={Qiuming Luo and Yuebing Li and Feng Li and Chang Kong},
      year={2026},
      eprint={2602.07768},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.07768}, 
}
```

---

## 📝 License

This project is licensed under the MIT License.
