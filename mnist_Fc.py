import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

from MNIST.raw.mnist import load_mnist
from torch.utils.data import DataLoader
from torch.nn.modules.activation import LeakyReLU

batch_size = 32
epochs = 30
learning_rate = 0.001

# 获取数据
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_train = torchvision.datasets.MNIST(root='./',train=True,download=True,transform=transform)
data_test  = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=transform)

data_train = DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True,num_workers=4, drop_last=True)
data_test  = DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

# 构建网络
class Net(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):					# 初始化网络结构
        super(Net, self).__init__()    	# 多继承需用到super函数
        self.layer = nn.Sequential(
            # 全连接层实现手写数字识别的功能
            nn.Linear(28 * 28, 1000), nn.Tanh(),
            nn.Linear(1000, 1000),    nn.Tanh(),
            nn.Linear(1000, 100),     nn.Tanh(),
            nn.Linear(100, 10),       nn.Tanh(),
            nn.Softmax()
        )
       
    # 正向传播过程
    def forward(self, x):  
        # out = x.reshape(x.size(0), -1) # 训练时打开网络
        out = self.layer(x)              # 训练时将x改为out
        
        return out

# 创建网络的实例
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.load_state_dict(torch.load('model_Fc.pkl'))

print(device)
print(torch.cuda.device_count())

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 0.0
num_batchs = len(data_train)

train_losses  = []
train_counter = []
test_losses   = []
test_counter  = []

def train():
    
    model.train()
    losses = 0.0
    train_loss = 0.0
    for iteration, (images, labels) in enumerate(data_train):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        train_loss += loss.item()
        #save acc
        if((iteration + 1) % 50 == 0):
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}%'.format(enpoch + 1, epochs, iteration + 1, num_batchs, losses / 50))
            train_losses.append(losses / 50)
            train_counter.append((iteration * batch_size) + ((enpoch - 1) * len(data_train.dataset)))
            losses = 0.0

def test():
    test_correct_num = 0
    for _,(data,target) in enumerate(data_test):
        data = data.to(device)
        target = target.to(device)
        output = model(data)# 正向传播得到预测值
        _, pred = torch.max(output, 1)
        test_correct_num += torch.sum(pred==target).item()
    print()
    print("Test Epoch:{}\t right_num: {}\t acc:{:.2f}".format(enpoch + 1, test_correct_num, test_correct_num/100.))
    print()

def opencv_development():
    print()


if __name__ == '__main__':
    for enpoch in range(epochs):
        train()
        test()
    # torch.save(model.state_dict(), 'model_Fc.pkl')
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('training num')
    plt.ylabel('loss descent')
    plt.show()

# 测试
# if __name__ == '__main__':
    # model.eval()
    # (_, _), (x_test, _) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

    # data = x_test[0]
    # x = data.reshape(-1, 784)
    # x =torch.from_numpy(x)
    # x = x.to(device)
    # # data = data.view(data.size(0), -1)
    # print(x.dtype)
    # out_data = model(x)
    # print(out_data)
    # # print(data, '\n', x.shape)