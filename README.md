# An-NAS-system-for-semantic-segmentation
所需环境（win，linux均可）：
python                            3.7（及以上）
pandas                            1.2.4
numpy                             1.19.5
torch                             1.7.1
torchvision                       0.8.2
Pillow                            9.0.1
matplotlib                        3.2.2
使用方法：
首先将文件夹所有模块下载下来
1.如果直接使用预训练的模型做测试，可以从此处（https://drive.google.com/drive/folders/1IynmiHDzKfhkZL_Oc41tI_iCC7gVImLt?usp=sharing）下载预训练模型（分别是Unet,Segnet文件夹），将这两个文件夹放于文件夹checkpoint(需要新建，并存放于与models文件夹平行的位置)中，运行Software.py文件即可出来前端页面：（1）选择需要测试的模型（Unet/Segnet）（2）从本地选择RGB图片（3）点击"Translated_Image"按钮，即可完成对本地RGB图像的分割任务
2.如果需要训练自己的数据集，将RGB数据集存放于文件夹src中，将对应的labeled数据集存放于label文件夹中，将这两个文件夹放在文件夹data/train（新建data文件夹，在其中建train文件夹，src和label文件夹在train里，把data文件夹放在与models平行的位置），之后分别运行train_U.py和train_Seg.py即可完成对Unet和Segnet的训练。训练完毕后，可以用Software.py可视化测试，方法同1.



