import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle
import math
import os
#################################################################################


class DataLoader:
    def __init__(self):
        
        self.current_batch = 0
        self.batch_size = 0
        self.batch_num = 0
        
        #data file 5개인데 이렇게 해도 ㄱㅊ??
        with open('./dataset/data_batch_' + str(1), 'rb') as fo:
            batch1 = pickle.load(fo, encoding='latin1')
            self.x_list = batch1['data']
            self.y_list = batch1['labels']
            
        with open('./dataset/data_batch_' + str(2), 'rb') as fo:
            batch2 = pickle.load(fo, encoding='latin1')
            x_list = batch2['data']
            self.x_list = np.append(self.x_list, x_list, axis=0)
            y_list = batch2['labels']
            self.y_list = np.append(self.y_list, y_list, axis=0)
            
        with open('./dataset/data_batch_' + str(3), 'rb') as fo:
            batch3 = pickle.load(fo, encoding='latin1')
            x_list = batch3['data']
            self.x_list = np.append(self.x_list, x_list, axis=0)
            y_list = batch3['labels']
            self.y_list = np.append(self.y_list, y_list, axis=0)
            
        with open('./dataset/data_batch_' + str(4), 'rb') as fo:
            batch4 = pickle.load(fo, encoding='latin1')
            x_list = batch4['data']
            self.x_list = np.append(self.x_list, x_list, axis=0)
            y_list = batch4['labels']
            self.y_list = np.append(self.y_list, y_list, axis=0)
            
        with open('./dataset/data_batch_' + str(5), 'rb') as fo:
            batch5 = pickle.load(fo, encoding='latin1')
            x_list = batch5['data']
            self.x_list = np.append(self.x_list, x_list, axis=0)
            y_list = batch5['labels']
            self.y_list = np.append(self.y_list, y_list, axis=0)
                        
            
            x_np = np.reshape(self.x_list,[-1,3,32,32])
            self.x = torch.from_numpy(x_np)
            self.x = self.x.float()
            
            y_np = np.asarray(self.y_list, dtype=np.int)
            y_torch = torch.from_numpy(y_np)
            y_np = np.reshape(self.y_list, [-1, 1])
            self.y = torch.from_numpy(y_np)
            self.y = self.y.float()
            self.y_onehot = torch.zeros(len(y_torch), y_torch.max()+1).scatter_(1, y_torch.unsqueeze(1), 1.)
            self.y = y_torch.long()
            self.data_size = len(y_np)
            print(self.data_size)
            
            
    def set_batch_size(self, size):
        """
        batch size를 정해줍니다.
        
        :param size:
        :return:
        """
        self.batch_size = size

    def get_batch_num(self):
        """
        batch의 수를 리턴합니다.

        :return: batch 수
        """
        self.batch_num = math.ceil(self.data_size / self.batch_size)
        return self.batch_num

    def get_batch(self):
        """
        전체 trainig 데이터에서 set_batch_size 에서 정한 batch size만큼만 데이터를 리턴합니다.
        이 떄 어디까지 리턴을 했는지를 저장하고 있다가 이전에 리턴한 데이터 다음 데이터부터 리턴을 합니다.
        나머지 데이터가 batch size보다 작을경우에는 남은 데이터만 리턴하고 어디까지 리턴했는지를 초기화 합니다.

        :return: batch
        """
        length = self.batch_size
        current_batch = self.current_batch
        
        if current_batch + length <= self.data_size :
            data_return = self.x[ current_batch : current_batch + length ]
            label_return = self.y_onehot[ current_batch : current_batch + length ] ##
            #label_return = self.y[ current_batch : current_batch + length ]
            self.current_batch += length
        else:
            data_return = self.x[ current_batch : -1 ]
            label_return = self.y_onehot[ current_batch : -1 ]
            #label_return = self.y[ current_batch : -1 ] ##
            self.current_batch = 0
            
        return data_return, label_return
        
    def reset(self):
        """
        어디까지 리턴했는지를 초기화 합니다.

        :return:
        """
        self.current_batch = 0

#################################################################################
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256 x 4 x 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #512 x 2 x 2
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(1024, 1)
            nn.Linear(128, 10)
        )
    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
        
###############################################################################

class Model:
    """
    모델의 cost 는 MSE를 사용합니다.
    """
    def __init__(self):
        self.net = Net()
        self.criterion = nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss()
        self.ismodelsaved = False

    def set_learning_rate(self, learning_rate):
        """
        learning rate를 정해줍니다.

        :param learning_rate:
        :return:
        """
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate , momentum=0.9)

    def train_step(self, x, y):
        """
        트레이닝 한 스템을 진행하는 함수로 input 데이터와 label 데이터를 입력으로 받아
        gradient와 cost를 계산하여 모델의 파라피터들을 업데이트 한 후 cost를 리턴합니다.

        :param x: input
        :param y: label
        :return: 현재 input에 대한 update 전의 cost
        """
        self.optimizer.zero_grad()
        
        outputs = self.net(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        
        self.optimizer.step()
        self.ismodelsaved = False # model을 saved 하고 load하고 다시 training 했을 때 새로운 모델은 저장이 안된 것으로 표시
        return loss
    
    def inference(self, x):
        """
        인풋에 대한 inference 결과를 계산 후 리턴합니다.

        :param x: input
        :return: 모델의 output
        """
        return self.net(x)

    def save(self):
        """
        모델의 파라미터들을 저장합니다.
        """
        ####와 하나도 모르겠다....
        torch.save(self.net.state_dict(), 'model.pt')
        self.ismodelsaved = True
        
    def restore(self):
        """
        save함수에서 저장한 모델의 파라미터를 restore합니다.

        :return: restore 되었으면 True아니면 False
        """
        if self.ismodelsaved == True:
            # model = TheModelClass(*args, **kwargs)
            self.net.load_state_dict(torch.load( 'model.pt'))
            self.net.eval()
        
        return self.ismodelsaved
        
        
        
        
