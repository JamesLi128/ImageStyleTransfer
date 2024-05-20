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
parser.add_argument("--style_name", default="VanGogh")
parser.add_argument("--style_weight", default=1e5, type=float)
parser.add_argument("--content_weight", default=1, type=float)
parser.add_argument("--tv_weight", default=1e-6, type=float)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

n_epochs = 1
batch_size = 4
lr = 0.001
device = torch.device(args.device)
img_folder = "unlabeled2017/"
style_name = args.style_name
style_img_path = '/home/jamesl/StyleTransfer/rst/content/' + style_name + ".jpg"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

coco_dataset = COCODataset(img_folder, transform)
coco_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size, shuffle=True)

style_img = Image.open(style_img_path).convert("RGB")
style_img = transform(style_img).unsqueeze(0).to(device)

backbone = Modified_CNN(models.vgg16(pretrained=True), [3, 8, 15, 22])
backbone.to(device)

style_feat_map_ls = backbone(style_img)

ImageTrans_model = ImageTranformNet()
ImageTrans_model.to(device)
optimizer = optim.Adam(ImageTrans_model.parameters(), lr=lr)
loss_fn = StyleTransferLoss(style_feat_map_ls, content_weight=args.content_weight, style_weight=args.style_weight, tv_weight=args.tv_weight)

best_loss = float("inf")
for epoch in range(n_epochs):
    for i, batch_imgs in enumerate(tqdm(coco_loader)):
        batch_imgs = batch_imgs.to(device)
        transformed_imgs = ImageTrans_model(batch_imgs)
        transformed_feat_map_ls = backbone(transformed_imgs)
        data_feat_map_ls = backbone(batch_imgs)
        loss, content_loss, style_loss, tv_loss = loss_fn(transformed_feat_map_ls, data_feat_map_ls, transformed_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, Style Loss: {style_loss.item()}, Content Loss: {content_loss.item()}, TV Loss: {tv_loss.item()}")
            if loss < best_loss:
                best_loss = loss
                torch.save(ImageTrans_model.state_dict(), f"{style_name}_best_model.pth")
            if i % 500 == 0:
                subprocess.call(["python", "test.py", "--style_name", style_name])
            