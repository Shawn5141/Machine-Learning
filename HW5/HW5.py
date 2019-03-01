import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.utils.data as Data
from collections import Counter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

def Loadfile(trainingfile,valfile):
	def Preprocessing(filename):
		with open(filename,'r',encoding='utf8') as f:
			print(filename)
			data=pd.read_csv(filename, header=None, sep='\t', quoting=3, compression='gzip')
			train_data=data.iloc[:,-1]
			train_target=data.iloc[:,0]

			col_con=np.squeeze(data.iloc[:,-1].values.reshape(1,-1))
			total=[]
			k=0
			for index in train_data:                     ########## construct entire character
				k+=1
				_index=list(index)
				total+=_index
			dic1=Counter(total)
			special_token=[]
			g=open("special_token",'w',encoding="utf-8")
			i=0
			j=0

			for key,val in dic1.items():
				if val<10:
					i+=1
					special_token.append(key)           ########### Save the special token
					g.write(key)
					g.write(' ')
				else:
					j+=1
			for key in special_token:
				if key in dic1:
					del dic1[key]                       ###########Construct a dic1 to save vocab

			dic1['<S>']=k
			dic1['</S>']=k
			l=0
			target_dict={}
			for index in train_target:
				target_dict[index]=target_dict.setdefault(index,0)+1
			for key,val in target_dict.items():
				target_dict[key]=l
				l+=1
			print(target_dict)
			print("Size of Vocabulary for "+filename,len(dic1))
			print("Percetage of out of Vocabulary=",i/(len(dic1)+i) )
			
			return dic1,train_data,train_target,target_dict
		
			
	Train_vocab,train_data,train_target,tr_target_dict=Preprocessing(trainingfile)
	Val_vocab,val_data,val_target,te_target_dict=Preprocessing(valfile)

	return Train_vocab,train_data,train_target,Val_vocab,val_data,val_target,te_target_dict

def Perplexity(Train_vocab,Val_vocab,val_data):
		Combine=[]
		Cross_entro=1/len(val_data)*np.log(len(val_data))
		for index,val in Val_vocab.items():
			if Train_vocab[index]:
				Cross_entro-=1/len(val_data)*np.log(Train_vocab[index]/len(Train_vocab))
			else:
				Cross_entro-=1/len(val_data)

		#for index in Combine:
		Proplexity=2**Cross_entro
		print("Cross_entropy,Proplexity",Cross_entro,Proplexity)

def CNN(train_data,train_target,Train_vocab,target_dict):
	print(len(train_data),len(train_target))
	def Preprocess(train_data,train_target,Train_vocab,savefile,target_dict):
		length=[] #Record the length of sequence
		data=[]
		for seq in train_data:
			length.append(len(list(seq)))
		print(len(train_data))
		_new_data=[]
		k=open(savefie,'w')
		dic2=Train_vocab
		v=0
		for key,val in dic2.items():
			val=v
			v+=1
			dic2[key]=val
			
		for seq in train_data:
			new_data=[]
			#print((np.max(length)-len(list(seq))))
			seq=list(seq)+['</S>']*(np.max(length)-len(list(seq)))
			for s in seq:
				new_data.append(Train_vocab[s])
			k.write(str(new_data))
			k.write('\n')

			_new_data.append(new_data)
	#Preprocess(train_data,train_target,Train_vocab,"wordvector_test.txt")
	def Read_word_vector(filename,Train_vocab):
		v=0
		character2id={}
		for key,val in Train_vocab.items():    ######## Language to 
			val=v
			v+=1
			character2id[val]=key

		r=open(filename,'r')
		_line=[]
		with open(filename,'r') as fp:  
			for seq_lengths, lines in enumerate(fp):
				line=lines.strip('[]\n').split(',')
				line=[int(i) for i in line]
				_line.append(line)

		#
		seq_tensor = Variable(torch.zeros((seq_lengths+1,len(line))).long())
		for idx,value in enumerate(_line):
			seq_tensor[idx:len(line)]=torch.LongTensor(value)
		
		embeds = torch.nn.Embedding(seq_lengths, 15)
		Embedding=embeds(seq_tensor)
		return torch.unsqueeze(Embedding,1)
	def Convolution_layer(Embedding,train_target):
		BATCH_SIZE=100
		learning_rate=0.001
		num_epochs = 10
		class ConvNet(nn.Module):
			def __init__(self):
				super(ConvNet, self).__init__()
				self.layer1 = nn.Sequential(
					nn.Conv2d(1, 32, kernel_size=15, stride=1, padding=7),    #32 channel (15-15+2*7)/1+1=15 #(157-15+14)/1+1=157
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=2, stride=2))                    #32 channel (15+1)/2+1=9, (157+1)/2+1=80  
				self.layer2 = nn.Sequential(
					nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=1),   #64 channel (9-9+2)/2+1=2 #(157-9+2)/2+1=76
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=2, stride=2))                    #(2-2)/2+1 =1    #(76-2)/2=36            
				self.drop_out = nn.Dropout()
				self.fc1 = nn.Linear(1 * 36 * 64, 1000)
				self.fc2 = nn.Linear(1000, 9)
			### Wout =(Win-F+2P)/S+1
			def forward(self,x):
				conv1=self.layer1(x)
				conv2=self.layer2(conv1)
				fc_input=conv2.view(conv2.size(0),-1)
				fc1=self.fc1(fc_input)
				fc2=self.fc2(fc1)
				return fc2

		def save_checkpoint(model, state, filename):
			model_is_cuda = next(model.parameters()).is_cuda
			model = model.module if model_is_cuda else model
			state['state_dict'] = model.state_dict()
			torch.save(state,filename)
		#Batch 
		_target=[]
		for target in train_target:
			_target.append(target_dict[target])

		x_label=torch.LongTensor(_target)
		print(x_label)
		print("fuck")

		torch_dataset = Data.TensorDataset(Embedding,x_label)

		loader = Data.DataLoader(
		dataset=torch_dataset,      # torch TensorDataset format
		batch_size=BATCH_SIZE,      # mini batch size
		shuffle=True,               # 
		num_workers=0,              # 
		)

		total_step = len(loader)
		if torch.cuda.is_available():
			model = ConvNet().cuda()
			criterion = nn.CrossEntropyLoss().cuda()
		model = ConvNet()
		# Loss and optimizer
		
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		criterion = nn.CrossEntropyLoss()
		loss_list = []
		acc_list = []
		for epoch in range(num_epochs):
			for i,(images, labels) in enumerate(loader):   ###Got an error on the dataloader
				images=Variable(images)
				labels=Variable(labels)
				print("fuck",labels)
				if torch.cuda.is_available():
					images=Variable(images).cuda()
					labels=Variable(labels).cuda()
				outputs = model(images)
				loss = criterion(outputs, labels)
				loss_list.append(loss.item())
				# Backprop and perform Adam optimisation
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


	Embedding=Read_word_vector('wordvector.txt',Train_vocab)
	Convolution_layer(Embedding,train_target)



if __name__ == '__main__':

	Train_vocab,train_data,train_target,Val_vocab,val_data,val_target,target_dict=Loadfile("train.tsv.gz","val.tsv.gz")
	#Perplexity(Train_vocab,Val_vocab,val_data)
	CNN(train_data,train_target,Train_vocab,target_dict)
