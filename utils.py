import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def MaxPool2AvgPool(model):
    features = model.features
    for i, layer in enumerate(features):
        if isinstance(layer, nn.MaxPool2d):
            features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

    model.features = features
    return model

class COCODataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.imgs = os.listdir(img_folder)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.imgs[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
class StyleLoss(nn.Module):
    def __init__(self, style_feat_map_ls):
        super(StyleLoss, self).__init__()
        self.style_feat_map_ls = style_feat_map_ls # n x 1 x c_j x h_j x w_j
        self.style_gram_matrices = [self.batch_gram_matrix(feat_map) for feat_map in style_feat_map_ls] # n, 1, c_j, c_j
        self.mse = nn.MSELoss()

    def batch_gram_matrix(self, feat_map):
        b, c, h, w = feat_map.size()
        feat_map = feat_map.view(b, c, h * w)
        gram_matrix = torch.bmm(feat_map, feat_map.transpose(1, 2))
        return gram_matrix.div(c * h * w)
    
    def forward(self, transformed_feat_map_ls):
        transformed_gram_matrices = [self.batch_gram_matrix(feat_map) for feat_map in transformed_feat_map_ls] # n x b x c_j x c_j
        loss = 0
        for i in range(len(self.style_gram_matrices)):
            loss += self.mse(transformed_gram_matrices[i], self.style_gram_matrices[i])
        return loss
    
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, transformed_feat_map_ls, data_feat_map_ls):
        loss = self.mse(transformed_feat_map_ls[1], data_feat_map_ls[1])
        return loss
    
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, transformed_imgs):
        h_tv = torch.sum(torch.abs(transformed_imgs[:, :, :, 1:] - transformed_imgs[:, :, :, :-1]))
        v_tv = torch.sum(torch.abs(transformed_imgs[:, :, 1:, :] - transformed_imgs[:, :, :-1, :]))
        return h_tv + v_tv
    
class StyleTransferLoss(nn.Module):
    def __init__(self, style_feat_map_ls, content_weight=1, style_weight=4e5, tv_weight=4e-5):
        super(StyleTransferLoss, self).__init__()
        self.style_loss = StyleLoss(style_feat_map_ls)
        self.content_loss = ContentLoss()
        self.tv_loss = TVLoss()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
    
    def forward(self, transformed_feat_map_ls, data_feat_map_ls, transformed_imgs):
        content_loss = self.content_loss(transformed_feat_map_ls, data_feat_map_ls)
        style_loss = self.style_loss(transformed_feat_map_ls)
        tv_loss = self.tv_loss(transformed_imgs)
        loss = self.content_weight * content_loss + self.style_weight * style_loss + self.tv_weight * tv_loss
        return loss, content_loss, style_loss, tv_loss
    
def load_random_imgs(img_folder, n_imgs, transform):
    img_paths = os.listdir(img_folder)
    idxs = np.random.choice(len(img_paths), n_imgs, replace=False)
    imgs = []
    for idx in idxs:
        img = Image.open(os.path.join(img_folder, img_paths[idx])).convert("RGB")
        img = transform(img)
        imgs.append(img)
    return torch.stack(imgs)


class OptLoss(nn.Module):
    def __init__(self, style_feat_map_ls, content_feat_map_ls, content_weight=0.1, style_weight=2e6, tv_weight=0) -> None:
        super(OptLoss, self).__init__()
        self.style_feat_map_ls = style_feat_map_ls
        self.content_feat_map_ls = content_feat_map_ls
        self.style_gram_ls = [self.gram_matrix(feat_map) for feat_map in style_feat_map_ls] # n x b x c_j x c_j
        self.mse = nn.MSELoss(reduction='sum')
        self.layer_weights = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    def content_loss(self, transformed_feat_map_ls):
        loss = self.mse(transformed_feat_map_ls[3], self.content_feat_map_ls[3])
        # loss = None
        # for (transformed_feat_map, content_feat_map) in zip(transformed_feat_map_ls, self.content_feat_map_ls):
        #     width, height = 256, 256
        #     if loss is None:
        #         loss = self.mse(transformed_feat_map, content_feat_map) / 2
        #     else:
        #         loss += self.mse(transformed_feat_map, content_feat_map) / 2
        return loss

    def gram_matrix(self, feat_map):
        b, c, h, w = feat_map.size()
        feat_map = feat_map.view(b, c, h * w)
        gram_matrix = torch.bmm(feat_map, feat_map.transpose(1, 2))
        return gram_matrix # b x c x c

    def style_loss(self, transformed_feat_map_ls):
        loss = None
        transformed_gram_ls = [self.gram_matrix(feat_map) for feat_map in transformed_feat_map_ls]
        for (transformed_gram, style_gram, weight) in zip(transformed_gram_ls, self.style_gram_ls, self.layer_weights):
            width, height = 256, 256
            if loss is None:
                loss = self.mse(transformed_gram, style_gram) * weight / ((2 * width * height) ** 2)
            else:
                loss += self.mse(transformed_gram, style_gram) * weight / ((2 * width * height) ** 2)
        return loss
    
    def tv_loss(self, transformed_imgs):
        h_tv = torch.sum(torch.abs(transformed_imgs[:, :, :, 1:] - transformed_imgs[:, :, :, :-1]))
        v_tv = torch.sum(torch.abs(transformed_imgs[:, :, 1:, :] - transformed_imgs[:, :, :-1, :]))
        return h_tv + v_tv

    def forward(self, transformed_feat_map_ls, transformed_imgs):
        content_loss = self.content_loss(transformed_feat_map_ls)
        style_loss = self.style_loss(transformed_feat_map_ls)
        tv_loss = self.tv_loss(transformed_imgs)
        loss = self.content_weight * content_loss + self.style_weight * style_loss + self.tv_weight * tv_loss
        return loss, content_loss, style_loss, tv_loss