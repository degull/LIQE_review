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

# BID 데이터셋의 이미지 파일 경로 및 메타데이터 파일 경로 설정
bid_image_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/ImageDatabase/'
metadata_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/DatabaseGrades.csv'

# BID 메타데이터 로드
metadata = pd.read_csv(metadata_path)
image_numbers = metadata['Image Number'].tolist()
mos_scores = metadata['Average Subjective Grade'].tolist()

print('###Image loading and testing###')

for image_num in image_numbers:
    img_file = f"DatabaseImage{int(image_num):04d}.JPG"  # 이미지 번호를 파일 이름으로 변환
    img_path = os.path.join(bid_image_path, img_file)

    if not os.path.exists(img_path):
        print(f"File {img_file} not found, skipping.")
        continue

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

    quality_prediction = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                         4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

    distortion_index = similarity_distortion.argmax(dim=1)
    scene_index = similarity_scene.argmax(dim=1)

    print(f'Image {img_file} is a photo of {scenes[scene_index]} with {dists[distortion_index]} artifacts, '
          f'which has a perceptual quality of {quality_prediction.item()} as quantified by LIQE.')

print('###Testing Complete###')
