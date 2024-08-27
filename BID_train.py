# BID 데이터셋에서 이미지와 주관적 등급(MOS)을 불러와서, CLIP 모델을 기반으로 한 이미지 품질 평가 모델을 학습

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
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, 'C:/Users/IIPL02/Desktop/LIQE/LIQE')

from utils import _preprocess2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BIDDataset(Dataset):
    def __init__(self, image_dir, metadata_path, preprocess):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = f"DatabaseImage{str(int(self.metadata.iloc[idx, 0])).zfill(4)}.JPG"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        
        # 이미지 크기 조정
        image = image.resize((224, 224))
        image = self.preprocess(image)

        mos = self.metadata.iloc[idx, 1]
        return image, mos

def train_model(model, text_features, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()

            # 이미지 특징 추출
            image_features = model.encode_image(inputs)

            # 텍스트 특징과 이미지 특징 사이의 유사도 계산
            logits_per_image = image_features @ text_features.t()

            # 유사도 평균을 통해 최종 예측값 생성
            predicted_mos = logits_per_image.mean(dim=1)

            # 손실 계산
            loss = criterion(predicted_mos, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model

if __name__ == '__main__':
    # 초기 설정
    seed = 20200626
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 모델 로드 및 전처리
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = model.float()  # 모델을 float32로 변환

    # BID 학습된 가중치 로드
    ckpt = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/pt/BID_fin.pt'  # BID 학습된 가중치 불러오기
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)

    # 텍스트 특징 생성
    dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 
             'contrast', 'lens', 'motion', 'diffusion', 'shifting', 'color quantization', 'oversaturation', 
             'desaturation', 'white with color', 'impulse', 'multiplicative', 'white noise with denoise', 'brighten', 
             'darken', 'shifting the mean', 'jitter', 'noneccentricity patch', 'pixelate', 'quantization', 
             'color blocking', 'sharpness', 'realistic blur', 'realistic noise', 'underexposure', 'overexposure', 
             'realistic contrast change', 'other realistic']

    scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
    qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

    joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                             in product(qualitys, scenes, dists)]).to(device).long()

    with torch.no_grad():
        text_features = model.encode_text(joint_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 데이터 로드
    image_dir = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/ImageDatabase/'
    metadata_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/IQA_Database/BID/DatabaseGrades.csv'
    dataset = BIDDataset(image_dir, metadata_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 손실 함수 및 옵티마이저
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 모델 학습
    trained_model = train_model(model, text_features, dataloader, criterion, optimizer, device, num_epochs=25)

    # 학습된 모델 저장
    torch.save(trained_model.state_dict(), 'C:/Users/IIPL02/Desktop/LIQE/LIQE/pt/BID_trained_on_BID_dataset_fin.pt')
