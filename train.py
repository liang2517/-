import torch.optim
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Lua
from load_data import *

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

train_data, test_data = load_dataset()
train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
lua = Lua()
lua.cuda()

loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()
learning_rate = 0.01
opt = torch.optim.Adam(lua.parameters(), lr=learning_rate)

total_train_step = 0

total_test_step = 0
epoch = 30
writer = SummaryWriter("other")



for i in range(epoch):
    print("------------第{} 轮训练开始-----------".format(i + 1))
    lua.train()
    for data in train_dataloader:
        data_x, data_y = handle_train_data(data)
        data_x = data_x.cuda()
        data_y = data_y.cuda()
        output1 = lua(data_x)
        loss = loss_fn(output1, data_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    lua.eval()
    total_acc = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            data_x, data_y = handle_test_data(data)
            data_x = data_x.cuda()
            data_y = data_y.cuda()
            output2 = lua(data_x)
            loss = loss_fn(output2, data_y)
            total_test_loss += loss.item()

    print("整体测试集上的LOSS：{}".format(total_test_loss))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

print("train:{}".format(output1))
print("test:{}".format(output2))
writer.close()

