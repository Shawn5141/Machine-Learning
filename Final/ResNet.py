from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets,models,transforms
import time
import os
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
	since = time.time()

	best_model_wts=model.state_dict()
	best_acc=0.0

	for epoch in range(num_epochs):
		print("Epoch{}/{}".format(epoch,num_epochs-1))
		print("-"*10)

		for phase in ['train','val']:
			if phase=='train':
				scheduler.step()
				model.train(True)

			if phase=='val':
				model.train(False)


			running_loss=0.0
			running_correct=0


			for data in dataloader[phase]:


				inputs,labels=data
				if use_gpu:
					inputs=Variable(inputs.cuda())
					labels=Variable(labels.cuda())

				else:
					inputs,labels=Variable(inputs),Variable(labels)

				optimizer.zero_grad()


				outputs=model(inputs)
				_,preds=torch.max(outputs.data,1)
				loss=criterion(outputs,labels)

				if phase =='train':
					loss.backward()
					optimizer.step()

				running_loss+=loss.item()*inputs.size(0)
				running_correct+=torch.sum(preds==labels.data)

			epoch_loss=running_loss/dataset_size[phase]
			epoch_acc=running_correct.double()/dataset_size[phase]
			#print("fuck",epoch_loss,epoch_acc)

			print('{}loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))

			if phase=='val' and epoch_acc>best_acc:
				best_acc=epoch_acc
				best_model_wts=model.state_dict()

	time_elasped=time.time()-since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elasped//60,time_elasped%60))
	print('best val Acc:{:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model

def visualize_model(model, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloader['val']):
			inputs = inputs.cuda()
			labels = labels.cuda()

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images//2, 2, images_so_far)
				ax.axis('off')
				ax.set_title('predicted: {}'.format(class_names[preds[j]]))
				imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return
		model.train(mode=was_training)



if __name__=='__main__':


	def imshow(inp, title=None):
		"""Imshow for Tensor."""
		inp = inp.numpy().transpose((1, 2, 0))
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		inp = std * inp + mean
		inp = np.clip(inp, 0, 1)
		plt.imshow(inp)
		if title is not None:
			plt.title(title)
		plt.pause(0.001)  # pause a bit so that plots are updated


	
	data_transform={
	'train':transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.465,0.406],[0.299,0.224,0.225])
		]),
	'val':transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.465,0.406],[0.299,0.224,0.225])    #how to decide
		]),
	}


	data_dir='/home/shawn/machine/Machine-Learning/Final/hymenoptera_data'
	image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transform[x]) for x in ['train','val']}
	dataloader={x:torch.utils.data.DataLoader(image_datasets[x],
		batch_size=4,
		shuffle=True,
		num_workers=4) for x in ['train','val']}
	dataset_size={x:len(image_datasets[x]) for x in ['train','val']}
	class_names=image_datasets['train'].classes

	# Get a batch of training data
	inputs, classes = next(iter(dataloader['train']))

	# Make a grid from batch
	out = torchvision.utils.make_grid(inputs)

	imshow(out, title=[class_names[x] for x in classes])


	use_gpu=torch.cuda.is_available()

	model_ft =models.resnet50(pretrained=True)
	num_ftrs=model_ft.fc.in_features
	model_ft.fc=nn.Linear(num_ftrs,2)

	if use_gpu:
		model_ft=model_ft.cuda()

	criterion=nn.CrossEntropyLoss()

	optimizer_ft=optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	model_ft=train_model(model=model_ft,
						optimizer=optimizer_ft,
						 criterion=criterion,
						 scheduler=exp_lr_scheduler,
						 num_epochs=25)
	visualize_model(model_ft)

	plt.ioff()
	plt.show()







