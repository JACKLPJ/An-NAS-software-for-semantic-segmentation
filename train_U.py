from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time
import numpy as np
from numpy import *
from data_loader.dataset import train_dataset, colorize_mask, fast_hist
from models.u_net import UNet
from models.seg_net import Segnet
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np
import cv2 as cv
parser = argparse.ArgumentParser(description='Training a UNet model')
parser.add_argument('--batch_size', type=int, default=4, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=2, help='equivalent to numclass')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool,default=True, help='enables cuda. default=True')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=2, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=256, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=256, help='scale image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--data_path', default='./data/train1', help='path to training images')
parser.add_argument('--outf', default='./checkpoint/Unet', help='folder to output images and model checkpoints')
parser.add_argument('--save_epoch', default=1, help='path to save model')
parser.add_argument('--test_step', default=300, help='path to val images')
parser.add_argument('--log_step', default=1, help='path to val images')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
    os.makedirs(opt.outf + '/model/')
except OSError:
    pass
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
cudnn.benchmark = True

print(opt)
print("Random Seed: ", opt.manual_seed)

train_datatset_ = train_dataset(opt.data_path, opt.size_w, opt.size_h, opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers)

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net = UNet(opt.input_nc, opt.output_nc)

if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net = nn.DataParallel(net)


###########   LOSS & OPTIMIZER   ##########
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#原始, weight_decay=0)
#optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1)
###########   GLOBAL VARIABLES   ###########
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
initial_image = Variable(initial_image)
#print(len(initial_image))
# print(initial_image)
# print(np.array(initial_image))
# print(np.array(initial_image).squeeze(0))
# print(np.array(initial_image.squeeze(0)))
# test_image = Image.open(test_image_path).convert('RGB')
# test_image=test_image.resize((256, 256),Image.ANTIALIAS) 
#test_image.resize_(initial_image_.size()).copy_(initial_image_)
# print('Operating...')
# img = input_transform(test_image)
# img = img.unsqueeze(0)
# img = Variable(img)


#sharpen_op = np.array([[-1/2, -1/2, -1/2], [-1/2, 4, -1/2], [-1/2, -1/2, -1/2]], dtype=np.float32)
#sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
#sharpen_image = cv.filter2D(prediction, -1, sharpen_op)

#print(initial_image)
#initial_image = torch.tensor(initial_image[opt.batch_size,:,:,:].astype(np.float32))
#img = torch.tensor(sharpen_image[None,:,:,:].astype(np.float32))
#pred_image = model(img)
# predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
# prediction = predictions[0]
# predictions_color = colorize_mask(prediction)


semantic_image = Variable(semantic_image)

if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()
    criterion = criterion.cuda()

if __name__ == '__main__':
    
    log = open('./checkpoint/Unet/train_Unet_log.txt', 'w')
    start = time.time()
    net.train()
    hist = np.zeros((opt.output_nc, opt.output_nc))
    for epoch in range(1, opt.niter+1):
        loader = iter(train_loader)
        for j in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = loader.next()
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            # for  i in range(len(initial_image)):
            #     sharpen_op = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]], dtype=np.float32)
            #     #sharpen_op = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            #     initial_image[i] = torch.tensor(cv.filter2D(initial_image[i].cpu().numpy().astype(np.uint8), cv.CV_32F, sharpen_op)).cuda()
            #     initial_image[i] = torch.tensor(cv.convertScaleAbs(initial_image[i].cpu().numpy())).cuda()
           # initial_image = torch.tensor(initial_image.astype(np.float32)).cuda()
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred = net(initial_image)
            # initial_image = initial_image.view(-1)
            # semantic_image_pred = semantic_image_pred.view(-1)

            ### loss ###
            # from IPython import embed;embed()
            assert semantic_image_pred.size()[2:] == semantic_image.size()[1:]
            loss = criterion(semantic_image_pred, semantic_image.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### evaluate ###
            predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts = semantic_image.data[:].squeeze_(0).cpu().numpy()
            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=opt.output_nc)
            train_acc = np.diag(hist).sum() / hist.sum()

            ########### Logging ##########
            if j% opt.log_step == 0:
                print('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                      (epoch, opt.niter, j, len(train_loader) * opt.batch_size, loss.item(), train_acc))
                log.write('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                          (epoch, opt.niter, j, len(train_loader) * opt.batch_size, loss.item(), train_acc))
            if j % opt.test_step == 0:
                gt = semantic_image[0].cpu().numpy().astype(np.uint8)
                gt_color = colorize_mask(gt)
                predictions = semantic_image_pred.data.max(1)[1].squeeze_(1).cpu().numpy()
                prediction = predictions[0]
                predictions_color = colorize_mask(prediction)
                width, height = opt.size_w, opt.size_h
                save_image = Image.new('RGB', (width * 2, height))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(predictions_color, box=(1 * width, 0 * height))
                save_image.save(opt.outf + '/epoch_%03d_%03d_gt_pred.png' % (epoch, j))

        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()