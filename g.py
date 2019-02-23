import torch
import torch.utils.data as Data
import os
import torch
import torchvision
import torch.utils.data as Data
import numpy as np
from ProgressBar1 import ProgressBar
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from sys import exit


EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
torch.manual_seed(1) 
BATCH_SIZE = 10



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20*20, 100),
            nn.Tanh(),
            nn.Linear(100, 25),

        )
        self.decoder = nn.Sequential(
            nn.Linear(25, 100),
            nn.Tanh(),
            nn.Linear(100, 20*20),
            nn.Tanh(), 
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def LoadDir(dirname):
    imgs = []
    process_bar = ProgressBar(len(os.listdir(dirname)),'Reading ok')
    for imgname in os.listdir(dirname):
        process_bar.show_process()
        img = Image.open(os.path.join(dirname, imgname))
        img = img.convert('LA')  # conver to grayscale
        img = img.resize([20, 20])
        img = np.squeeze(np.array(img)[:, :, 0])
        imgs.append(img)
        
    imgs=np.array(imgs)
    data=imgs.reshape(imgs.shape[0],400)
    data=np.divide(data,255)
    data = torch.from_numpy(data).float()
    torch_dataset = Data.TensorDataset(data,data)

    loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 
    num_workers=2,              # 
)
    return np.array(imgs),loader

def PCA_transform(img):
    pca=PCA(n_components=25)
    img=img.reshape(len(img),400)
    newData=pca.fit(img)
    data_reduced = np.dot(img - pca.mean_, pca.components_.T)
    nComp = 25
    Xhat = np.dot(pca.transform(img)[:,:nComp], pca.components_[:nComp,:])+np.mean(img, axis=0)
    print(np.sum((img-Xhat)**2)/len(img)/400)
    return np.sum((img-Xhat)**2)/len(img)/400

def AutoEncode(train,test,LR,re_error):

    def PCA_transform(img):
        pca=PCA(n_components=25)
        img=img.reshape(len(img),400)
        newData=pca.fit(img)
        data_reduced = np.dot(img - pca.mean_, pca.components_.T)
        nComp = 25
        Xhat = np.dot(pca.transform(img)[:,:nComp], pca.components_[:nComp,:])+np.mean(img, axis=0)
        print(np.sum((img-Xhat)**2)/len(img)/400)
        


    if torch.cuda.is_available():
        autoencoder = AutoEncoder().cuda()
        loss_func = nn.MSELoss().cuda()
    else:
        autoencoder = AutoEncoder()
        loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    process_bar = ProgressBar(EPOCH,'Training ok')
    for epoch in range(EPOCH):
        process_bar.show_process()
        for step, (x, b_label) in enumerate(train):
            
            b_x = x.view(-1, 20*20)   # batch x, shape (batch, 28*28)
            b_y = x.view(-1, 20*20)   # batch y, shape (batch, 28*28)
            if torch.cuda.is_available():
                b_x=Variable(b_x).cuda()
                b_y=Variable(b_y).cuda()
            encoded, decoded = autoencoder(b_x)
            #loss=loss_func(decoded,b_y)
            loss = loss_func(torch.mul(decoded,255), torch.mul(b_y,255))      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            #print('Epoch: ', epoch, '| train loss: %.4f' % (loss/400))
    Loss=[]
    Special_test=[]
    Special_num=[]
   
    for step, (x, b_label) in enumerate(test):
        
        b_x = x.view(-1, 20*20)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 20*20)   # batch y, shape (batch, 28*28)
        if torch.cuda.is_available():
            b_x=Variable(b_x).cuda()
            b_y=Variable(b_y).cuda()
        encoded, decoded = autoencoder(b_x)
        #loss=loss_func(decoded,b_y)
        loss = loss_func(torch.mul(decoded,255), torch.mul(b_y,255))      # mean square error   
        
        if loss < re_error:
            #Special_test.append([step,b_x.cpu().detach().numpy()])
            Special_test.append(step)
            
        #print(loss)
        Loss.append(loss.cpu().detach().numpy())
    Loss=np.array(Loss).reshape(-1,len(Loss))
    #print(Loss.shape[1])
    Loss=np.sum(np.squeeze(Loss))/Loss.shape[1]
    print(Loss)
    print(Special_test)
        #if epoch % 10 == 0:
            #pic = to_img(output.cpu().data)
            #save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    #torch.save(model.state_dict(), './sim_autoencoder.pth')

if __name__ == '__main__':
    #train_imgs,train_loader = LoadDir('galaxy/train')
    #test_imgs,test_loader=LoadDir('galaxy/test')
    train_imgs,train_loader = LoadDir('galaxy/psudo1')
    test_imgs,test_loader=LoadDir('galaxy/test')
    re_error_val=PCA_transform(train_imgs)
    re_error_test=PCA_transform(test_imgs)
    AutoEncode(train_loader,test_loader,LR,re_error_test)
    

    source = "/home/shawn/machine/Machine-Learning/galaxy/test/"
    target = "/home/shawn/machine/Machine-Learning/galaxy/Special_graph/"
    i=0
    g=open('number_for_special.txt','r')
    lines=g.read()
    lines=lines.strip("[]\n").split(", ")
    for filename in os.listdir('galaxy/test'):
        for l in lines:
            print(l,i)
            if int(l)==i:
                print("low")
                shutil.copy(source+str(filename),target+str(filename))
        i+=1
            
                

        



 
