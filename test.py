import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
import torchvision.models as models
from utils import *
from model import *
import argparse



def test(model_path, img_folder, device, style_name):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    random_imgs = load_random_imgs(img_folder, 5, transform).to(device)
    model = ImageTranformNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    transformed_imgs = model(random_imgs.to(device))
    # print(transformed_imgs.shape)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    random_imgs = random_imgs.cpu() * std + mean

    output = torch.cat([random_imgs.cpu(), transformed_imgs.cpu()], dim=-1)
    # print(output.shape)

    # output = output * std + mean
    output = output.clamp(0, 1)
    output_images = []
    for i in range(output.shape[0]):
        img = transforms.ToPILImage()(output[i])
        output_images.append(img)
        img.save(f'tmp/{style_name}_img_{i}.jpg')
    return output_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_name', default="VanGogh")
    args = parser.parse_args()
    style_name = args.style_name
    model_path = style_name + '_best_model.pth'
    img_folder = 'unlabeled2017/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(model_path, img_folder, device, style_name)
        
