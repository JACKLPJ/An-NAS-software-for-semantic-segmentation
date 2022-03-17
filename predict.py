from __future__ import print_function
import os
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from models.u_net import UNet
from models.seg_net import Segnet
from data_loader.dataset import input_transform, colorize_mask
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
#model = Segnet(3,2)
#model_path = './checkpoint/Segnet/model/netG_final.pth'
"""
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--test_image_path', type=str, default = '/home/sia1/TF SS/Semantic-segmentation-master-1/Semantic-segmentation-master/data/train/src/120.png')
#parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()
"""
model = UNet(3, 2)
model_path = './checkpoint/Unet/model/netG_1.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
test_image_path = './data/train/src/120.png'#args.test_image_path#'./data/train/src/120.png'

lena = mpimg.imread(test_image_path) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print(lena.shape) #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
img = input_transform(test_image)
img = img.unsqueeze(0)
img = Variable(img)
pred_image = model(img)
predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
prediction = predictions[0]
predictions_color = colorize_mask(prediction)
predictions_color.show()
