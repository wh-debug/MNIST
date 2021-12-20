import torch
import os
import cv2
from torch._C import device
from torch.nn.modules.activation import ReLU
from tqdm import *
from torch.nn.modules.conv import LazyConv1d
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Scale

camer = cv2.VideoCapture(0, cv2.CAP_DSHOW)
image_size = 28

batch_size = 16
epochs = 50
learning_rate = 0.00095

# 训练标志
VGGMODEL = False
train = 2

# 获取数据
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_train = torchvision.datasets.MNIST(root='./',train=True,download=True,transform=transform)
data_test = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=transform)

data_train = DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True,num_workers=4, drop_last=True)
data_test = DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)
print(len(data_train)) # 938张

# 构建网络
class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.layer1 = nn.Sequential(
            # in_channels：图像的通道数，out_channels: 卷积产生的通道数，kernel_size: 卷积核尺寸， stride： 卷积步长， padding： 填充操作， padding_mode: padding模式，
            # dilation： 扩张工作， 控制kernel的间距，默认是1, groups： group参数的作用是控制分组卷积， bias： 为真，则在输出中添加一个可学习的偏差。默认：True。
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # 输出为32*28*28
            nn.ReLU(),# 经过一个relu函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为32*14*14，最大池化层
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  # 输出为16*10*10
            nn.ReLU(),# 经过一个relu函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为7*7*64
        )
        self.drop_out = nn.Dropout()
        self.flatten1 = nn.Linear(7 * 7 * 64, 1000)
        self.flatten2 = nn.Linear(1000, 10)
    # 正向传播过程
    def forward(self, x):  
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.flatten1(out)
        out = self.flatten2(out)
        return out
# 构建网络
class VGG16(nn.Module):
    def __init__(self):						# 初始化网络结构
        super(VGG16, self).__init__()    	# 多继承需用到super函数
        self.vgg16 = models.vgg16(pretrained=True)
        #后面的全链接的层
        self.classfsify = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        input = self.vgg16(input)
        input = self.classfsify(input)
        return input

# 创建网络的实例
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if VGGMODEL:
    model = VGG16().to(device)
    # 阻止优化器更新权重
    for param in model.children():
        param.requires_grad = False
    print(model)
else:
    model = LeNet().to(device)
    model.load_state_dict(torch.load('model.pkl'))



print(device)
print(torch.cuda.device_count())

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 0.0
writer = SummaryWriter(log_dir='logs')
num_batchs = len(data_train)
if train == 0:
    for enpoch in range(epochs):
        model.train()
        losses = 0.0
        writer.add_scalar(tag='train_loss', scalar_value=train_loss / num_batchs, global_step=enpoch)
        train_loss = 0.0
        for iteration, (images, labels) in enumerate(data_train):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # train_loss += loss.item()
            # background 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            train_loss += loss.item()
            # print(iteration, images.size(), labels.size(), loss.item())
            #save acc
            if((iteration + 1) % 50 == 0):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}%'.format(enpoch + 1, epochs, iteration + 1, num_batchs, losses / 50))
                losses = 0.0
    writer.close()
elif train == 1:
    model.eval()
    test_correct_num = 0
    # print(len(data_test.dataset))
    with torch.no_grad():   # 不更新参数
        for epoch in range(0, epochs):
            test_correct_num = 0
            for batch_idx,(data,target) in enumerate(data_test):
                data = data.to(device)
                target = target.to(device)
                output = model(data)# 正向传播得到预测值
                _, pred = torch.max(output, 1)
                test_correct_num += torch.sum(pred==target).item()
                # print(pred)
                # print(target)
                # print(test_correct_num)
            print("Test Epoch:{}\t right_num: {}\t acc:{:.2f}".format(epoch + 1, test_correct_num, test_correct_num/100.))
else:
    model.eval()
    camer = cv2.VideoCapture('rtsp://admin:12345@192.168.3.142:8554/live')
    with torch.no_grad():
        while camer.isOpened():
            ret, frame = camer.read()
            keypressed = cv2.waitKey(1)
            print('键盘按下的键是：',  keypressed)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
            # gray = cv2.resize(gray, (28, 28))
            cv2.imshow('image', gray)
            # 27应该是Esc的编码
            if keypressed == 27:
                break
        camer.release()
        cv2.destroyAllWindows()

# torch.save(model.state_dict(), 'model.pkl')


