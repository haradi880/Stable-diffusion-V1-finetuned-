# 🎨 Pokémon Stable Diffusion Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Generate anime-style Pokémon characters using text descriptions.**
> This project utilizes **LoRA (Low-Rank Adaptation)** to fine-tune Stable Diffusion v1.5 on a dataset of 35,000+ Pokémon images.

### Generated Example
![Generated Pokémon](output/result2.png)
*Prompt: "cute electric pokemon, anime style, soft lighting"*

---

## 🌟 Quick Links

| Resource | Link | Description |
| :--- | :--- | :--- |
| **🎯 Model** | [**Kaggle Model**](https://www.kaggle.com/models/haradibots/stable-diffusion-v1-finetuned/PyTorch/default/1/) | Final weights (Epoch 5) |
| **📊 Dataset** | [**35K Pokémon Data**](https://www.kaggle.com/datasets/saranga7/35k-pokemon-and-text-descriptions/versions/1) | Images & Captions |
| **📓 Notebook** | [**Training Log**](https://www.kaggle.com/code/haradibots/bkl-new-way-to-work-what-1/edit/run/290183288) | Code execution |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/pokemon-stable-diffusion-finetuning.git](https://github.com/yourusername/pokemon-stable-diffusion-finetuning.git)
cd pokemon-stable-diffusion-finetuning

# Install dependencies
pip install -r requirements.txt

```

### 2. Inference (Generate Images)

Use the provided `inference.py` script to generate new Pokémon.

```python
from inference import PokemonGenerator

# Load the fine-tuned model
generator = PokemonGenerator("models/offline_model_epoch_4")

# Generate a Pokémon image
image = generator.generate(
    prompt="cute electric pokemon, anime style, soft lighting",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=30,
    guidance_scale=7.5
)
image.save("my_pokemon.png")

```

### 3. Training

To train the model yourself (requires GPU):

```bash
python train.py --epochs 5 --batch-size 32 --data-dir "data/"

```

---

## 📁 Project Structure

```text
pokemon-stable-diffusion-finetuning/
├── train.py              # Training script (Accelerate/Peft)
├── inference.py          # Inference/generation class
├── requirements.txt      # Python dependencies
├── outputs/              # Generated images
│   └── offline_result.png
├── models/               # Model checkpoints
└── data/                 # Dataset information

```

---

## 📊 Model Details

| Parameter | Value |
| --- | --- |
| **Base Model** | Stable Diffusion v1.5 |
| **Method** | LoRA (Rank 16) |
| **Resolution** | 512×512 |
| **Target Modules** | `to_q`, `to_k`, `to_v`, `to_out.0` |
| **Training Time** | ~4-6 hours (Kaggle P100) |

---

## 🛠️ Requirements

* **Python:** 3.10+
* **GPU:** NVIDIA 8GB VRAM (Minimum) / 16GB+ (Recommended for training)
* **Disk Space:** ~10GB

**Core Dependencies:**

* `torch>=2.0.0`
* `diffusers>=0.21.0`
* `transformers>=4.35.0`
* `accelerate>=0.24.0`
* `peft>=0.6.0`


## 👥 Authors & Acknowledgments

**Aditya Haradibots**

* [LinkedIn](https://www.linkedin.com/in/aditya-haradibots/)
* [Kaggle](https://www.kaggle.com/code/haradibots/)

*This project is for educational purposes. Weights based on Stable Diffusion v1.5.*

```

```
