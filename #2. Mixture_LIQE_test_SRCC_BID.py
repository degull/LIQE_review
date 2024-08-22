import torch
import numpy as np
import clip
from utils import _preprocess2
import random
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F
import os
import pandas as pd
import scipy.stats

import sys
sys.path.insert(0, 'C:/Users/IIPL02/Desktop/LIQE/LIQE')

from utils import _preprocess2

ImageFile.LOAD_TRUNCATED_IMAGES = True

dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion', 'shifting',
         'color quantization', 'oversaturation', 'desaturation', 'white with color', 'impulse', 'multiplicative',
         'white noise with denoise', 'brighten', 'darken', 'shifting the mean', 'jitter', 'noneccentricity patch',
         'pixelate', 'quantization', 'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
         'underexposure', 'overexposure', 'realistic contrast change', 'other realistic']

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

# BID 데이터셋의 이미지 파일 경로 설정
image_folder = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/ImageDatabase/'
metadata_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/DatabaseGrades.csv'

metadata = pd.read_csv(metadata_path)
image_files = [f"DatabaseImage{str(int(num)).zfill(4)}.JPG" for num in metadata['Image Number'].tolist()]  # 이미지 파일 이름 생성
mos_scores = metadata['Average Subjective Grade'].tolist()

print('###Image loading and testing###')

predicted_scene_quality_scores = []
predicted_distortion_quality_scores = []
quality_predictions = []

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)

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
    for i in range(num_patch):
        sel[i] = sel_step * i
    sel = sel.long()
    I_patches = I_patches[sel, ...]
    I_patches = I_patches.to(device)

    print(f'Processing {img_file}...')

    with torch.no_grad():
        logits_per_image, _ = do_batch(I_patches.unsqueeze(0), joint_texts)

    logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists))
    logits_quality = logits_per_image.sum(3).sum(2)
    similarity_scene = logits_per_image.sum(3).sum(1)
    similarity_distortion = logits_per_image.sum(1).sum(1)

    # Quality prediction
    quality_prediction = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                         4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
    quality_predictions.append(quality_prediction.cpu().item())

    # Scene + Quality prediction
    scene_weight = 0.4
    quality_weight = 0.6
    logits_scene_quality = (scene_weight * similarity_scene + quality_weight * logits_quality.mean(dim=1, keepdim=True)).mean(dim=1)
    scene_quality_prediction = logits_scene_quality.cpu().item()

    # Distortion + Quality prediction
    logits_distortion_quality = (scene_weight * similarity_distortion + quality_weight * logits_quality.mean(dim=1, keepdim=True)).mean(dim=1)
    distortion_quality_prediction = logits_distortion_quality.cpu().item()

    predicted_scene_quality_scores.append(scene_quality_prediction)
    predicted_distortion_quality_scores.append(distortion_quality_prediction)

# SRCC 계산
srcc_quality = scipy.stats.spearmanr(quality_predictions, mos_scores)[0]
srcc_scene_quality = scipy.stats.spearmanr(predicted_scene_quality_scores, mos_scores)[0]
srcc_distortion_quality = scipy.stats.spearmanr(predicted_distortion_quality_scores, mos_scores)[0]

print(f"### SRCC (Quality): {srcc_quality:.4f} ###")
print(f"### SRCC (Scene + Quality): {srcc_scene_quality:.4f} ###")
print(f"### SRCC (Distortion + Quality): {srcc_distortion_quality:.4f} ###")
