import os
import re
import torch
import torchvision.transforms as transforms
from PIL import Image

white_img = torch.ones(3, 256, 256)
transforms.ToPILImage()(white_img).save("rst/white.jpg")