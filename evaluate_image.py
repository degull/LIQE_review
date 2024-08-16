import sys
import importlib.util

spec = importlib.util.spec_from_file_location("liqe", "C:/Users/IIPL02/Desktop/LIQE/LIQE/liqe.py")
liqe_module = importlib.util.module_from_spec(spec)
sys.modules["liqe"] = liqe_module
spec.loader.exec_module(liqe_module)

LIQE = liqe_module.LIQE
from PIL import Image
from torchvision import transforms
import torch

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/pt/LIQE.pt'
    liqe = LIQE(ckpt, device)

    # 실제 이미지 불러오기
    image_path = 'C:/Users/IIPL02/Desktop/LIQE/LIQE/data/I02_01_03.png'
    image = Image.open(image_path).convert("RGB")

    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    x = preprocess(image).unsqueeze(0).to(device)  # 배치 차원 추가

    q, s, d = liqe(x)

    # 결과 출력
    print(f"Quality: {q.item()}")
    print(f"Scene: {s}")
    print(f"Distortion: {d}")
