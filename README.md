______________________________________________________________________

<div align="center">

# PAND: Prompt-Augmented Neighborhood Distillation for Fine-Grained Visual Recognition

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://pytorchlightning.ai/">
  <img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://hydra.cc/">
  <img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>
<a href="https://github.com/ashleve/lightning-hydra-template">
  <img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray">
</a>

</div>

---

## 📋 Description

**PAND** (Prompt-Augmented Neighborhood Distillation) is an advanced knowledge distillation framework that combines:
- **CoOp** (Context Optimization): Learnable prompt tuning for vision-language models
- **VL2Lite**: Task-specific knowledge distillation from large VLMs to lightweight networks
- **NLRD** (Neighborhood Logits Relation Distillation): Preserving neighborhood relationships in logit space

This framework enables efficient transfer of knowledge from large CLIP-based models to compact student networks (ResNet18, MobileNetV2) for fine-grained visual recognition tasks.

---


## 🚀 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd PAND
```

### 2. Create Environment
```bash
conda create -n pand_env python=3.9
conda activate pand_env
```

### 3. Install Dependencies
```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 4. Install OpenCLIP (for CLIP teacher)
```bash
pip install open-clip-torch
```

---

## 📁 Dataset Preparation

### Supported Datasets
- CUB-200-2011
- FGVC Aircraft
- Oxford-IIIT Pet
- Stanford Dogs

### Setup Data Path
Update dataset paths in `configs/data/attributes/`:
```yaml
# Example: configs/data/attributes/0_CUB_200_2011.yaml
data_root: /path/to/CUB_200_2011
```

Or create symbolic links:
```bash
ln -s /your/dataset/path /data/lvta/datasets/vl2lite_datasets
```

---

## 🎓 Training Pipeline

### Stage PSD: CoOp Prompt Learning

#### Option 1: Train CoOp with Full Images (Memory Intensive ~15GB GPU)
```bash
python src/train_coop.py \
  --config configs/data/attributes/0_CUB_200_2011.yaml \
  --batch_size 32 \


---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory during CoOp Training**
```bash
# Solution 1: Use pre-extracted features
python src/extract_image_features.py  # Extract first
python src/train_coop_cub_features.py  # Then train

# Solution 2: Reduce batch size
--batch_size 16
```

**2. NLRD Loss Causing NaN**
```bash
# Check batch size (too small batches can cause issues)
data.batch_size=128  # Recommended minimum: 64

# Reduce NLRD weight
model.kd_criterion.nlrd_weight=0.5
```

**3. Different Sample Counts in t-SNE**
- This is caused by `drop_last=True` in DataLoader with different batch sizes
- The script automatically handles this by taking the minimum sample count

**4. ImportError for OpenCLIP**
```bash
pip install open-clip-torch
```

---

## 🛠️ Advanced Usage

### Custom Dataset
1. Implement dataset class in `src/data/components/kd_dataloader.py`
2. Create config in `configs/data/attributes/`
3. Add dataset attributes (class names, prompt template)

### Custom Student Model
```python
# In configs/model/coop_kd.yaml
student:
  model_name: "your_model"  # Must be in torchvision.models
  pretrained: true
```

### Resume Training
```bash
python src/train.py \
  ckpt_path=/path/to/checkpoint.ckpt \
  trainer.max_epochs=400  # Continue to epoch 400
```

---

## 📚 Key References

- **CoOp**: Learning to Prompt for Vision-Language Models (IJCV 2022)
- **VL2Lite**: Task-Specific Knowledge Distillation from Large VLMs (CVPR 2025)
- **NRKD**: Neighborhood Relation-based Knowledge Distillation (Pattern Recognition 2021)

---

## 🙏 Acknowledgments

- Built upon [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- CLIP models from [OpenCLIP](https://github.com/mlfoundations/open_clip)
- Thanks to PyTorch, PyTorch Lightning, and Hydra communities

---

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{pand2025,
  title={PAND: Prompt-Augmented Neighborhood Distillation for Fine-Grained Visual Recognition},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}

@inproceedings{jang2025vl2lite,
  title={VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks},
  author={Jang, Jinseong and Ma, Chunfei and Lee, Byeongwon},
  booktitle={CVPR},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
  model.net.student.model_name=resnet18 \
  trainer=ddp \
  trainer.devices=4 \
  trainer.max_epochs=300 \
  data.batch_size=128
```

#### 2. VL2Lite + CoOp (No NLRD)
```bash
python src/train.py \
  data/attributes=0_CUB_200_2011 \
  model=coop_kd \
  model.net.student.model_name=resnet18 \
  model.net.teacher.coop_text_features=/path/to/learned_text_features.pt \
  model.kd_criterion.use_coop=true \
  model.kd_criterion.use_nlrd=false \
  trainer=ddp \
  trainer.devices=4 \
  trainer.max_epochs=300 \
  data.batch_size=128
```

#### 3. Full Pipeline: CoOp + NLRD (Ours)
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

#### 4. NLRD Sensitivity Analysis
Test different λ (NLRD weight) values:
```bash
for lambda in 0 0.05 0.1 0.2 0.5 1.0; do
  python src/train.py \
    data/attributes=0_CUB_200_2011 \
    model=coop_kd \
    model.kd_criterion.nlrd_weight=$lambda \
    trainer=ddp \
    trainer.devices=4 \
    logger=csv \
    tags=["cub","nlrd","lambda$lambda"]
done
```

---

## 📊 Evaluation & Visualization

### Test Model
```bash
python src/eval.py \
  ckpt_path=/path/to/checkpoint.ckpt \
  data/attributes=0_CUB_200_2011
```

### Generate t-SNE Visualization
Compare three methods (w/o KD, VL2Lite, Ours):
```bash
# Edit checkpoint paths in scripts/tsne_compare.py
python scripts/tsne_compare.py
```

Output: `tsne_cub_compare.png` and individual plots

---

## ⚙️ Configuration Guide

### Model Configs
- `configs/model/kda.yaml`: VL2Lite baseline (no CoOp)
- `configs/model/coop_kd.yaml`: CoOp + KD framework

### Key Parameters

**CoOp Settings:**
- `n_ctx`: Number of learnable context tokens (default: 16)
- `ctx_init`: Context initialization (None for random)
- `csc`: Class-specific context (default: False)

**NLRD Settings:**
- `use_nlrd`: Enable NLRD loss
- `nlrd_k`: Number of neighbors (default: 3)
- `nlrd_lambda`: Temperature for neighbor selection (default: 1.0)
- `nlrd_weight`: Loss weight coefficient (default: 1.0)

**Training Settings:**
- `batch_size`: Batch size (128 recommended for 4 GPUs)
- `learning_rate`: 1e-4 for student, 0.002 for CoOp
- `temperature`: KD temperature (default: 2.0)

### Multi-GPU Training
```bash
# DDP with 4 GPUs
trainer=ddp trainer.devices=4

# Single GPU
trainer=gpu trainer.devices=1
```
---

## Configuration

- **trainer** configs in `configs/trainer/`  
- **data** configs in `configs/data/`  
- **model** configs in `configs/model/`  
- **experiment** configs in `configs/experiment/`  

Hydra allows combining or overriding these configs easily.

---

## Acknowledgments

Built upon the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).  
We thank open-source projects (PyTorch, Lightning, Hydra) that enable this work.

---

## License

This project is licensed under the [MIT License](LICENSE).
Please see the [LICENSE](LICENSE) file for details.
