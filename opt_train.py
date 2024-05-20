import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from utils import *
from model import *
from tqdm import tqdm
import warnings
import subprocess
import re
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=10000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--style_name", type=str, default="VanGogh")
parser.add_argument("--content_img_path", type=str, default="rst/content/river_bank.jpg")
parser.add_argument("--style_weight", type=float, default=1e5)
parser.add_argument("--content_weight", type=float, default=1)
parser.add_argument("--tv_weight", type=float, default=1e-6)
args = parser.parse_args()

n_epochs = args.iters
lr = 0.001
device = torch.device(args.device)
content_img_path = args.content_img_path
style_img_path = f"{args.style_name}.jpg"
style_name = args.style_name

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

style_img = Image.open(style_img_path).convert("RGB")
style_img = transform(style_img).unsqueeze(0).to(device)
content_img = Image.open(content_img_path).convert("RGB")
content_img = transform(content_img).unsqueeze(0).to(device)

backbone = Modified_CNN(MaxPool2AvgPool(models.vgg19(pretrained=True)), [1, 6, 11, 20, 29])
backbone.to(device)

style_feat_map_ls = backbone(style_img)
content_feat_map_ls = backbone(content_img)

Generated_Image = nn.Parameter(torch.randn(1, 3, 256, 256).to(device) * 0.3, requires_grad=True)
optimizer = optim.Adam([Generated_Image], lr=lr, betas=(0.5, 0.999), weight_decay=1e-3)
loss_fn = OptLoss(style_feat_map_ls, content_feat_map_ls, style_weight=args.style_weight, content_weight=args.content_weight, tv_weight=args.tv_weight)

tensor2pil = transforms.Compose([
    transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44)),
    transforms.ToPILImage()
])

print("Start Training")

for epoch in tqdm(range(n_epochs)):
    transformed_feat_map_ls = backbone(Generated_Image)
    transformed_feat_map_ls = [feat_map for feat_map in transformed_feat_map_ls]
    loss, content_loss, style_loss, tv_loss = loss_fn(transformed_feat_map_ls, Generated_Image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}, TV Loss: {tv_loss.item()}")
        if (epoch+1) % 200 == 0:
            tensor2pil(Generated_Image[0].cpu()).save(f"opt_img/Generated_{epoch+1}.jpg")
