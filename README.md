<div align="center">

# REASON: Rule-based Explainable Action via Symbolic cONcepts

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)]()

**A Neurosymbolic Framework for Concept-Driven Logical Reasoning in Skeleton-Based Human Action Recognition**

<!-- [Paper]() | [Project Page]() | [Demo]() -->

</div>


---

## 🎯 Overview

**REASON** (**R**ule-based **E**xplainable **A**ction via **S**ymbolic c**ON**cepts) is a neurosymbolic framework for skeleton-based human activity recognition (HAR) that formulates action recognition as logical reasoning over learnable motion concepts and integrates concept-grounded representations with differentiable first-order logic for end-to-end training.

![teaser](teasor.png)

### Abstract

Skeleton-based human activity recognition has achieved strong empirical performance, yet most existing models remain black boxes and difficult to interpret. In this work, we introduce a neurosymbolic formulation of skeleton-based HAR that reframes action recognition as concept-driven first-order logical reasoning over motion primitives. Our framework bridges representation learning and symbolic inference by grounding first-order logic predicates in learnable spatial and temporal motion concepts. 

Specifically, we employ a standard spatio-temporal skeleton encoder to extract latent motion representations, which are then mapped to interpretable concept predicates via a spatio-temporal concept decoder that explicitly separates pose-centric and dynamics-centric abstractions. These concept predicates are composed through differentiable first-order logic layers, enabling the model to learn human-readable logical rules that govern action semantics. To impose semantic structure on the learned concepts, we align skeleton representations with LLM-derived descriptions of atomic motion primitives, establishing a shared conceptual space for perception and reasoning. 

Extensive experiments on **NTU RGB+D 60/120** and **NW-UCLA** demonstrate that our approach achieves competitive recognition performance while providing explicit, interpretable explanations grounded in logical structure. Our results highlight neurosymbolic reasoning as an effective paradigm for interpretable spatio-temporal action understanding.

---

## ✨ Key Features

- 🧠 **Neurosymbolic Reasoning**: Combines neural perception with symbolic logical reasoning
- 🔍 **Interpretable Concepts**: Learns human-readable motion primitives (pose-centric and dynamics-centric)
- 🔗 **Differentiable Logic**: End-to-end training with first-order logic layers
- 🎯 **LLM-Grounded Concepts**: Aligns learned concepts with language model descriptions
- 📊 **Strong Performance**: Competitive accuracy on major benchmarks (NTU 60/120, NW-UCLA)
- 🔬 **Explainable Predictions**: Provides logical explanations for action classifications

---

## 🏗️ Architecture

[Add teaser figure here]

The framework consists of three main components:

1. **Spatio-Temporal Encoder**: Extracts latent motion representations from skeleton sequences
2. **Concept Decoder**: Maps representations to interpretable spatial and temporal concepts
3. **Logic Reasoning Layer**: Composes concepts through differentiable first-order logic rules

---

## 🚀 Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/reason.git
cd reason

# Create conda environment
conda create -n reason python=3.10
conda activate reason

# Install dependencies
pip install -r requirements.txt

# Install torchlight
pip install -e torchlight
```

### Required Packages

- PyTorch >= 2.4.0
- PyYAML
- tqdm
- wandb
- transformers
- timm
- huggingface_hub
- accelerate

---

## 📁 Data Preparation

We follow the data preparation pipeline from [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

### Download Datasets

#### NTU RGB+D 60 and 120

1. Request dataset from: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   - `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
3. Extract to `./data/nturgbd_raw`

#### NW-UCLA

1. Download from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Extract `all_sqe` to `./data/NW-UCLA`

### Directory Structure

Organize the data as follows:

```
data/
├── NW-UCLA/
│   └── all_sqe/
│       └── ... (raw data)
├── ntu/
├── ntu120/
└── nturgbd_raw/
    ├── nturgb+d_skeletons/      # from s001-s017
    │   └── ...
    └── nturgb+d_skeletons120/   # from s018-s032
        └── ...
```

### Generate Processed Data

**For NTU RGB+D 60:**

```bash
cd ./data/ntu

# Extract skeleton data for each performer
python get_raw_skes_data.py

# Remove corrupted skeletons
python get_raw_denoised_data.py

# Center-normalize skeletons
python seq_transformation.py
```

**For NTU RGB+D 120:**

```bash
cd ./data/ntu120

# Follow same steps as above
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

---

## 🎓 Training

### NTU RGB+D 60 Cross-Subject

```bash
python main.py \
  --config config/nturgbd-cross-subject/default.yaml \
  --work-dir ./work_dir/ntu60/xsub \
  --device 0
```

### NTU RGB+D 60 Cross-View

```bash
python main.py \
  --config config/nturgbd-cross-view/default.yaml \
  --work-dir ./work_dir/ntu60/xview \
  --device 0
```

### NTU RGB+D 120 Cross-Subject

```bash
python main.py \
  --config config/nturgbd120-cross-subject/default.yaml \
  --work-dir ./work_dir/ntu120/xsub \
  --device 0
```

### NTU RGB+D 120 Cross-Set

```bash
python main.py \
  --config config/nturgbd120-cross-set/default.yaml \
  --work-dir ./work_dir/ntu120/xset \
  --device 0
```

### NW-UCLA

```bash
python main.py \
  --config config/ucla/default.yaml \
  --work-dir ./work_dir/ucla \
  --device 0
```

### Multi-Stream Training

Train with different modalities (joint, bone, velocity):

```bash
# Joint stream
python main.py --config config/nturgbd-cross-subject/lst_joint.yaml

# Bone stream
python main.py --config config/nturgbd-cross-subject/lst_bone.yaml

# Joint velocity
python main.py --config config/nturgbd-cross-subject/lst_joint_vel.yaml

# Bone velocity
python main.py --config config/nturgbd-cross-subject/lst_bone_vel.yaml
```

---

## 📊 Evaluation

### Test Pre-trained Model

```bash
python eval.py \
  --config config/nturgbd-cross-subject/default.yaml \
  --weights ./work_dir/ntu60/xsub/best_model.pt \
  --device 0
```

### Ensemble Evaluation

Combine multiple streams for improved performance:

```bash
python eval.py \
  --config config/nturgbd-cross-subject/default.yaml \
  --weights ./work_dir/ntu60/xsub/joint.pt \
            ./work_dir/ntu60/xsub/bone.pt \
            ./work_dir/ntu60/xsub/joint_vel.pt \
            ./work_dir/ntu60/xsub/bone_vel.pt \
  --device 0
```


---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{reason2026,
                PAPER CURRENTLY UNDER REVIEW
}
```

---

## 🙏 Acknowledgments

- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) for data preparation pipelines.
- [Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN/tree/main) backbones and ablations.
- [GAP](https://github.com/MartinXM/GAP/tree/main) backbones and ablations.
- NTU RGB+D dataset providers
- NW-UCLA dataset providers

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<!-- 
## 📧 Contact

For questions or issues, please open an issue or contact [your.email@example.com](mailto:your.email@example.com). -->

---

<div align="center">

**[⬆ back to top](#reason-rule-based-explainable-action-via-symbolic-concepts)**

</div>

