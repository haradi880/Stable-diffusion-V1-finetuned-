# !pip install -q diffusers transformers accelerate einops torchvision
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from datasets import load_dataset

# Best for Kaggle's 8-hour limit
from peft import LoraConfig
from diffusers import DiffusionPipeline

# Load pretrained model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none"
)
pipe.unet.add_adapter(lora_config)

import os
import re
from PIL import Image

IMG_DIR = "/kaggle/input/35k-pokemon-and-text-descriptions/train"
OUT_DIR = "/kaggle/working/clean_data"
os.makedirs(OUT_DIR, exist_ok=True)

pairs = []

def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9,.\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for file in os.listdir(IMG_DIR):
    if not file.endswith(".jpg"):
        continue
    
    idx = file.replace(".jpg", "")
    img_path = os.path.join(IMG_DIR, f"{idx}.jpg")
    txt_path = os.path.join(IMG_DIR, f"{idx}.txt")

    if not os.path.exists(txt_path):
        continue

    try:
        caption = open(txt_path, "r", encoding="utf-8").read().strip()
        caption = clean_caption(caption)
        if len(caption.split()) < 3:
            continue

        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(OUT_DIR, f"{idx}.jpg"))
        open(os.path.join(OUT_DIR, f"{idx}.txt"), "w").write(caption)

        pairs.append(idx)
    except:
        continue

print("Clean pairs:", len(pairs))

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PokemonDataset(Dataset):
    def __init__(self, root, tokenizer, size=512):
        self.root = root
        self.tokenizer = tokenizer
        self.ids = [f.replace(".jpg", "") for f in os.listdir(root) if f.endswith(".jpg")]

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        image = Image.open(f"{self.root}/{id_}.jpg").convert("RGB")
        caption = open(f"{self.root}/{id_}.txt").read()

        image = self.transform(image)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0],
        }
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer
unet = pipe.unet
vae = pipe.vae
text_encoder = pipe.text_encoder

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import os
from tqdm import tqdm

BATCH_SIZE = 32
GRAD_ACCUM = 1
LR = 1e-5
EPOCHS = 5
DEVICE = "cuda"

dataset = PokemonDataset("/kaggle/working/clean_data", tokenizer)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

optimizer = AdamW(unet.parameters(), lr=LR)
noise_scheduler = pipe.scheduler

unet.train()
vae.eval()
text_encoder.eval()

global_step = 0

for epoch in range(EPOCHS):
    pbar = tqdm(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(
            DEVICE, dtype=torch.bfloat16
        )
        input_ids = batch["input_ids"].to(DEVICE)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            text_embeds = text_encoder(input_ids).last_hidden_state

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (latents.size(0),),
            device=DEVICE
        )

        noisy_latents = noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        pbar.set_description(
            f"Epoch {epoch} | Loss {loss.item():.4f}"
        )

    # Save checkpoint each epoch
    SAVE_DIR = f"/kaggle/working/offline_model_epoch_{epoch}"
    unet.save_pretrained(f"{SAVE_DIR}/unet")
    vae.save_pretrained(f"{SAVE_DIR}/vae")
    text_encoder.save_pretrained(f"{SAVE_DIR}/text_encoder")
    noise_scheduler.save_pretrained(f"{SAVE_DIR}/scheduler")
    
tokenizer.save_pretrained(f"{SAVE_DIR}/tokenizer")
import shutil

MODEL_DIR = "/kaggle/working/offline_model_epoch_4"
ZIP_PATH = "/kaggle/working/pokemon_sd_offline_epoch_4.zip"

shutil.make_archive(
    ZIP_PATH.replace(".zip", ""),
    "zip",
    MODEL_DIR
)

print("ZIP created at:", ZIP_PATH)
