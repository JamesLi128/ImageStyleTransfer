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

parser = argparse.ArgumentParser()
parser.add_argument("--renew_content", default=False, type=bool, help="Whether to renew the content images")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--iters", default=2000, type=int, help="Number of iterations")
parser.add_argument("--style_weight", default=3e5, type=float, help="Weight of style loss")
parser.add_argument("--content_weight", default=1, type=float, help="Weight of content loss")
parser.add_argument("--tv_weight", default=1e-6, type=float, help="Weight of total variation loss")                    
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")
parser.add_argument("--style_name", default=None, type=str, help="Name of the style image")
parser.add_argument("--content_save_path", default="rst/content/", type=str, help="Path to save content images")
parser.add_argument("--style_path", default='rst/corpped_style/', type=str, help="Path to the style images")
args = parser.parse_args()

warnings.filterwarnings("ignore")

def compare_once(args, style_name):
    args.iters
    lr = args.lr
    device = torch.device(args.device)
    style_img_path = args.style_path + style_name + ".jpg"
    model_path = style_name + "_best_model.pth"
    content_save_path =args.content_save_path

    # Transform_Net = ImageTranformNet()
    # Transform_Net.load_state_dict(torch.load(model_path))
    # Transform_Net.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tensor2pil = transforms.Compose([
        transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44)),
        transforms.ToPILImage()
    ])

    style_img = Image.open(style_img_path).convert("RGB")
    style_img = transform(style_img).unsqueeze(0).to(device)
    # tensor2pil(style_img[0].cpu()).save(f"rst/corpped_style/corpped_{style_name}.jpg")

    idx2img_name = {}
    
    if args.renew_content:
        content_random_imgs = load_random_imgs("unlabeled2017/", 5, transform).to(device)
        for i, img in enumerate(content_random_imgs):
            img = tensor2pil(img.cpu())
            img.save(content_save_path + f"content_img_{i}.jpg")
            idx2img_name[i] = f"content_img_{i}"
    else:
        img_path_ls = os.listdir(content_save_path)
        pattern = re.compile(r"\.jpg")
        img_name_ls = [pattern.sub("", img_path) for img_path in img_path_ls]
        img_path_ls = [os.path.join(content_save_path, img_path) for img_path in img_path_ls]
        imgs = []
        for i, path in enumerate(img_path_ls):
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            imgs.append(img)
            idx2img_name[i] = img_name_ls[i]
        content_random_imgs = torch.cat(imgs, dim=0)

    for i, content_img in enumerate(content_random_imgs):
        tensor2pil(content_img.cpu()).save(f"rst/corpped_content/corpped_{idx2img_name[i]}.jpg")

    backbone = Modified_CNN(MaxPool2AvgPool(models.vgg19(pretrained=True)), [1, 6, 11, 20, 29])
    backbone.to(device)

    style_feat_map_ls = backbone(style_img)
    content_feat_map_ls = backbone(content_random_imgs)

    Generated_Imgs = nn.Parameter(torch.randn(5, 3, 256, 256).to(device) * 0.3)
    optimizer = optim.Adam([Generated_Imgs], lr=lr, betas=(0.5, 0.999), weight_decay=1e-3)
    loss_fn = OptLoss(style_feat_map_ls, content_feat_map_ls, style_weight=args.style_weight, content_weight=args.content_weight, tv_weight=args.tv_weight)


    # print("Generating with Pre-trained Tranform Net")
    # Transformed_imgs = Transform_Net(content_random_imgs)
    # Transformed_imgs = Transformed_imgs.clamp(0, 1)
    # for i, img in enumerate(Transformed_imgs):
    #     img = transforms.ToPILImage()(img.cpu())
    #     img.save(f"rst/transformed/{style_name}_transformed_{idx2img_name[i]}.jpg")

    for i in tqdm(range(args.iters), desc=f"Style: {style_name}"):
        transformed_feat_map_ls = backbone(Generated_Imgs)
        loss, content_loss, style_loss, tv_loss = loss_fn(transformed_feat_map_ls, Generated_Imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"Iter: {i+1}, Loss: {loss.item()}, Style Loss: {style_loss.item()}, Content Loss: {content_loss.item()}, TV Loss: {tv_loss.item()}")
            for j, img in enumerate(Generated_Imgs):
                img = tensor2pil(img.cpu())
                img.save(f"rst/generated/{style_name}_generated_{idx2img_name[j]}.jpg")


def compare_models(args):
    style_name = args.style_name
    if style_name is None:
        style_name_ls = ["VanGogh", "Scream", "Monet", "Picasso", "Kandinsky"]
        for style_name in style_name_ls:
            compare_once(args, style_name)
    else:
        compare_once(args, style_name)
    
if __name__ == "__main__":
    compare_models(args)