import torch
import numpy as np
import clip
from utils import _preprocess2
import random
from itertools import product
from PIL import Image, ImageFile
import os
import pandas as pd
import scipy.stats
import torch.nn.functional as F

import sys
sys.path.insert(0, 'C:/Users/IIPL02/Desktop/LIQE/LIQE')

from utils import _preprocess2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Distortion, Scene, Quality 속성 정의
dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 
         'contrast', 'lens', 'motion', 'diffusion', 'shifting', 'color quantization', 'oversaturation', 
         'desaturation', 'white with color', 'impulse', 'multiplicative', 'white noise with denoise', 'brighten', 
         'darken', 'shifting the mean', 'jitter', 'noneccentricity patch', 'pixelate', 'quantization', 
         'color blocking', 'sharpness', 'realistic blur', 'realistic noise', 'underexposure', 'overexposure', 
         'realistic contrast change', 'other realistic']

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

preprocess2 = _preprocess2()

def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, logits_per_text = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image = logits_per_image.mean(1)
    logits_per_text = logits_per_text.mean(2)

    # Softmax 적용을 통해 정규화
    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image, logits_per_text


seed = 20200626
num_patch = 15

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
ckpt = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/pt/LIQE.pt'  # LIQE 사전 학습된 가중치 불러오기
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists)]).to(device)

# KonIQ-10k 데이터셋의 이미지 파일 경로 설정
koniq10k_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/koniq-10k/1024x768/'
metadata_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/koniq-10k/meta_info_KonIQ10kDataset.csv'

# KonIQ-10k 메타데이터 로드
metadata = pd.read_csv(metadata_path)
image_files = metadata['image_name'].tolist()
mos_scores = metadata['MOS'].tolist()

predicted_quality_scores = []
predicted_scene_scores = []
predicted_distortion_scores = []

print('### Image loading and testing ###')

for i, img_file in enumerate(image_files):
    img_path = os.path.join(koniq10k_path, img_file)

    I = Image.open(img_path)
    I = preprocess2(I)
    I = I.unsqueeze(0)
    n_channels = 3
    kernel_h = 224
    kernel_w = 224

    if (I.size(2) >= 1024) | (I.size(3) >= 1024):
        step = 48
    else:
        step = 32
    I_patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)
    sel_step = I_patches.size(0) // num_patch
    sel = torch.zeros(num_patch)
    for j in range(num_patch):
        sel[j] = sel_step * j
    sel = sel.long()
    I_patches = I_patches[sel, ...]
    I_patches = I_patches.to(device)

    print(f'Processing {img_file}...')

    with torch.no_grad():
        logits_per_image, _ = do_batch(I_patches.unsqueeze(0), joint_texts)

    logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists))

    # Quality prediction
    logits_quality = logits_per_image.sum(3).sum(2)
    quality_prediction = (logits_quality * torch.arange(1, len(qualitys) + 1, device=device).float()).sum(dim=1)
    predicted_quality_scores.append(quality_prediction.cpu().item())

    # Scene prediction
    scene_prediction = logits_per_image.sum(3).argmax(dim=2).float().mean(dim=1)
    predicted_scene_scores.append(scene_prediction.cpu().item())

    # Distortion prediction
    distortion_prediction = logits_per_image.sum(2).argmax(dim=2).float().mean(dim=1)
    predicted_distortion_scores.append(distortion_prediction.cpu().item())

# SRCC 계산
srcc_quality = scipy.stats.spearmanr(predicted_quality_scores, mos_scores)[0]

# 학습 가능한 가중치
alpha_scene = torch.nn.Parameter(torch.tensor(0.5, device=device))
alpha_distortion = torch.nn.Parameter(torch.tensor(0.5, device=device))

# SRCC for Quality + Scene (with learned weight)
combined_quality_scene_scores = [(alpha_scene * q + (1 - alpha_scene) * s).item() for q, s in zip(predicted_quality_scores, predicted_scene_scores)]
srcc_quality_scene = scipy.stats.spearmanr(combined_quality_scene_scores, mos_scores)[0]

# SRCC for Quality + Distortion (with learned weight)
combined_quality_distortion_scores = [(alpha_distortion * q + (1 - alpha_distortion) * d).item() for q, d in zip(predicted_quality_scores, predicted_distortion_scores)]
srcc_quality_distortion = scipy.stats.spearmanr(combined_quality_distortion_scores, mos_scores)[0]

# SRCC for Quality + Scene + Distortion
combined_quality_scene_distortion_scores = [(alpha_scene * s + alpha_distortion * d + (1 - alpha_scene - alpha_distortion) * q).item()
                                            for q, s, d in zip(predicted_quality_scores, predicted_scene_scores, predicted_distortion_scores)]
srcc_quality_scene_distortion = scipy.stats.spearmanr(combined_quality_scene_distortion_scores, mos_scores)[0]

print(f"### SRCC (Quality): {srcc_quality:.4f} ###")
print(f"### SRCC (Quality + Scene): {srcc_quality_scene:.4f} ###")
print(f"### SRCC (Quality + Distortion): {srcc_quality_distortion:.4f} ###")
print(f"### SRCC (Quality + Scene + Distortion): {srcc_quality_scene_distortion:.4f} ###")

print('### Testing Complete ###')
