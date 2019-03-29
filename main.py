import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from torch.nn import functional
from torch.autograd import Variable
from data.dataset import DogCat
from models.ResNet34 import ResNet34
from matplotlib import pyplot as plt
import time


def train():
    # step1:加载数据
    train_data_root = "D:\\github\\DogVsCat_mianjin\\data\\data\\train\\train"
    train_data = DogCat(train_data_root, train=True)
    val_data = DogCat(train_data_root, train=False)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                             batch_size = 33,
                             shuffle = True,
                             num_workers =4 )
    val_dataloader = torch.utils.data.DataLoader(val_data,
                             batch_size = 32,
                             shuffle = True,
                             num_workers =4 )

    #step2:加载模型
    model = ResNet34().cuda()

    # step3:目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr =0.005,
                                 weight_decay=1e-4)

    # step4:训练
    loss_list=[]
    start = time.time()
    for epoch in range(30):
        # 共有11576个图片已知结果的图片，按照7：3分成训练集和验证集
        # 训练集有11576*0.7=8103张图片
        # 验证集有11576*0.3=3472张图片
        # 这里batch_size = 32，也就是每次送入32张图片进入网络进行训练，
        # num_workers =4，就是采用4个线程进行训练，加快训练速度
        # 所以每个epoch 循环8103/32=253次
        # 采用GTX1050Ti,batch_size=32,num_workers=4,GPU占用率为80%，训练一个epoch需要需要循环253次，用时386s
        # 采用GTX1050Ti,batch_size=32,num_workers=1,GPU占用率仍为80%，训练一个epoch需要循环253次，用时377s
        # 采用GTX1050Ti,batch_size=16,num_workers=1,GPU占用率为60%以下，训练一个epoch需要循环506次，用时819s
        # 采用GTX1050Ti,batch_size=33,num_workers=4,GPU占用率83%左右，训练一个epoch需要循环次245次，用时391s
        # 采用GTX1050Ti,batch_size=34及其以上,num_workers=4,GPU爆表了
        # 看来如何最大效率的利用GPU是一个需要考虑的问题
        # 这里使用了cuda()加速，也就是利用GPU来运算，相对于CPU,GPU能够并行的处理，大大加快了运行的速度
        for i,(data,label) in enumerate(train_dataloader):
            input = Variable(data).cuda()
            target = Variable(label).cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            # if i%10 == 9:
            #     print(i)
            print("epoch:%d,time%d"%(epoch,i), "time:", (time.time() - start), loss.cpu().detach().numpy())
        # 每训练一个周期就用验证集的数据测试一次，看准确率是否有提升
        # 预测每个batch的labels,将其与实际的labels对比，得到正确率
        correct = 0
        # for i,(data,label) in enumerate(val_dataloader):
        #     input = Variable(data).cuda()
        #     target = Variable(label).cuda()
        #     score = model(input)
        #     score = score.cpu().data
        #     correct += (score==target).sum()
        # print(correct)
        # loss_list.append(loss.numpy())
        # print("epoch:%d: %d,train loss:%d"%(epoch,i,loss.numpy))
    name = 'checkpoints/resnet34.pth'
    torch.save(model.state_dict(), name)
    plt.plot(loss,'r')
    plt.show()

def test():
    model = ResNet34().cuda()

def val():
    pass

def write_csv():
    pass

if __name__=='__main__':
    train()