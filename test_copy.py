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
import argparse     
import torchvision
import matplotlib.pyplot as plt
import cv2 as cv
#from util.visualizer import Visualizer                                                                                     #
#from utils import save_images
#model = Segnet(3,2)
#model_path = './checkpoint/Segnet/model/netG_final.pth'
print('*****************************************************')
parser = argparse.ArgumentParser(description='Put the path of Image into test_image_path')               #
parser.add_argument('--test_image_path',type=str,help='The variable which stores the path of Image')     #
parser.add_argument('--Model',type=str,help='The variable which chooses the model to be tested')     #
args = parser.parse_args()                                                                               #


model = args.Model#UNet(3, 2)
# model = UNet(3, 2)
model_path = './checkpoint/Unet/model/netG_final.pth'
if model=='Unet':
    model = UNet(3, 2)
    model_path = './checkpoint/Unet/model/netG_final.pth'
elif model=='Segnet':
    model = Segnet(3, 2)
    model_path = './checkpoint/Segnet/model/netG_final.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))                 
# test_image_path = './data/train1/src/WCDP_247.tif'
test_image_path = args.test_image_path                                                                   # 把传入的值赋给test_image_path变量

print(test_image_path)

"""
lena = mpimg.imread(test_image_path) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了,可以对它进行任意处理
print(lena.shape) #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
"""
test_image = Image.open(test_image_path).convert('RGB')
test_image=test_image.resize((256, 256),Image.ANTIALIAS) 
#test_image.resize_(initial_image_.size()).copy_(initial_image_)
print('Operating...')
img = input_transform(test_image)
img = img.unsqueeze(0)
img = Variable(img)
pred_image = model(img)
predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
prediction = predictions[0]
predictions_color = colorize_mask(prediction)
print(prediction)
#src = os.path.join(os.path.abspath(img_path), img) #原先的图片名字
dst = test_image_path.split('/')[-1]#os.path.join(os.path.abspath(img_path), 'E_' + img) #根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
#os.rename(predictions_color, dst) #重命名,覆盖原先的名字
#predictions_color.show()
#print(dst)
sharpen_op = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
#sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
#sharpen_image = cv.filter2D(prediction, -1, sharpen_op)
sharpen_image = cv.filter2D(np.array(prediction).astype(np.uint8), cv.CV_32F, sharpen_op)
sharpen_image = cv.convertScaleAbs(sharpen_image)
print(sharpen_image)
cv.imwrite('./Image_translated/'+dst.split('.')[0]+'.png',sharpen_image)

#cv.imshow("sharpen_image", sharpen_image)
#predictions_color.save('./Image_translated/'+dst.split('.')[0]+'.png')
#imgPath= './Image_translated'
#plt=predictions_color
#torchvision.utils.save_image(predictions_color, imgPath)
#plt.savefig('1.png')#第一个是指存储路径，第二个是图片名字
# 指定图片保存路径
"""
figure_save_path = "/home/sia1/TFSS/Semantic-segmentation-master-1/Semantic-segmentation-master/Image_translated"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path , dst))#第一个是指存储路径，第二个是图片名字
"""
#plt.show()/home/sia1/TFSS/Semantic-segmentation-master-1/Semantic-segmentation-master/Image_translated
#print('/home/sia1/TFSS/Semantic-segmentation-master-1/Semantic-segmentation-master/Image_translated/'+dst)
#plt.savefig('/home/sia1/TFSS/Semantic-segmentation-master-1/Semantic-segmentation-master/Image_translated/'+dst)
#torchvision.utils.save_image(predictions_color, imgPath)

#predictions_color.show()
"""
import cv2 as cv
import numpy as np

src = cv.imread("C:/Users/qqxd/Desktop/opencvcode/images/test.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

sharpen_op = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]], dtype=np.float32)
#sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpen_image = cv.filter2D(src, cv.CV_32F, sharpen_op)
sharpen_image = cv.convertScaleAbs(sharpen_image)
cv.imshow("sharpen_image", sharpen_image)

h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = sharpen_image
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", result)
# cv.imwrite("D:/result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
"""