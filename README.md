# An-NAS-software-for-semantic-segmentation
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
1.如果直接使用预训练的模型做测试，可以从此处（https://drive.google.com/drive/folders/1aJCfK8wG4XVCl5vvqrxmN3Swq1RQxW29?usp=sharing,   https://drive.google.com/drive/folders/1r3dn4mRVfc96-e0UZKEbEiWcmgS8g0qa?usp=sharing  ）  
下载预训练模型（分别是u,seg文件夹），将这两个文件夹中的文件分别拷贝到文件夹checkpoint/Unet/model和checkpoint/Segnet/model中，运行Software.py文件即可出来前端页面：
![f45ec938ce1be79bbac3508d08179ee](https://user-images.githubusercontent.com/42956088/156692085-6b7e2544-73f8-4d39-8215-7de5cee5f398.png)  
（1）选择需要测试的模型（Unet/Segnet）
![69b2a244bd2b8344571af74962b0165](https://user-images.githubusercontent.com/42956088/156692109-445a41a0-c046-41f2-8750-df8447b63b62.png)
（2）从本地选择RGB图片
![517ab48f77054e9bb8299647e6e46b9](https://user-images.githubusercontent.com/42956088/156692140-10e3b478-7f8d-4587-861d-fa0ca644f853.png)
（3）点击"Translated_Image"按钮，即可完成对本地RGB图像的分割任务
![99d54021d87de5599317e09aa694da3](https://user-images.githubusercontent.com/42956088/158728377-65a50f01-2a05-4de1-bd45-7c3dbd8f9551.jpg)
2.如果需要训练自己的数据集，将RGB数据集存放于文件夹src中，将对应的labeled数据集存放于label文件夹中，将这两个文件夹放在文件夹data/train（新建data文件夹，在其中建train文件夹，src和label文件夹在train里，把data文件夹放在与models平行的位置），之后分别运行train_U.py和train_Seg.py即可完成对Unet和Segnet的训练。训练完毕后，可以用Software.py可视化测试，方法同1.
