import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from resnet import MnistResNet

class RotEqCNN():
  def __init__(self, ensemble_num = 9, dataset='mnist', train_epoch = 6):
		self.ensemble_num = ensemble_num
		self.dataset = dataset
		self.rotate_angle = 360.0 / (self.ensemble_num * 1.0)
		self.start_angle=-270
		self.end_angle=0
		self.train_transform = []
		self.trainset = []
		self.trainloader = []
		self.model=[]
		self.optimizer=[]
		self.criterion = nn.CrossEntropyLoss()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.encoded_train_data = []
		self.encoded_train_label = []
		self.mlp = None
		self.X_test = None
		self.y_test = None
		self.train_epoch = train_epoch
	def get_dataset(self):
		means = deviations = [0.5]
		for i in range(self.ensemble_num):
			self.train_transform.append(transforms.Compose([transforms.RandomRotation([self.start_angle,self.end_angle]),transforms.ToTensor(),transforms.Normalize(means, deviations)]))
			start_angle=-270+self.rotate_angle*(i+1)
			end_angle=self.rotate_angle*(i+1)
		# add trainset
		for i in range(self.ensemble_num):
			self.trainset.append(torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=self.train_transform[i]))
		# add trainloader
		for i in range(self.ensemble_num):
			self.trainloader.append(torch.utils.data.DataLoader(self.trainset[i], batch_size=128,shuffle=True, num_workers=2))
	def init_models(self):
		# define models
		for i in range(self.ensemble_num):
			self.model.append(MnistResNet().to(self.device))
		# define optimizers
		for i in range(self.ensemble_num):
			self.optimizer.append(optim.SGD(self.model[i].parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4))
	def train(self):
		epoch_range = self.train_epoch
		# training loop + eval loop
		for ensemble_id in range(self.ensemble_num):
			running_loss = 0.0
			print("Loss of ensemble model", ensemble_id)
			for epoch in range(epoch_range):
			  for i, data in enumerate(self.trainloader[ensemble_id], 0):
				  # get the inputs
				  inputs, labels = data
		  #         print(labels.numpy().shape)

				  inputs, labels = inputs.to(self.device), labels.to(self.device)

				  # zero the parameter gradients
				  self.optimizer[ensemble_id].zero_grad()

				  # forward + backward + optimize
				  outputs = self.model[ensemble_id](inputs)
				  loss = self.criterion(outputs, labels)
				  loss.backward()
				  self.optimizer[ensemble_id].step()

				  # print statistics
				  running_loss += loss.item()
				  if i % 20 == 19:    # print every 2000 mini-batches
				      print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 2000))
				      running_loss = 0.0

		"""## Step 6: Form the encoded sets"""

		correct = 0
		for i in range(self.ensemble_num):
			with torch.no_grad():
				for data in self.trainloader[i]:
					images, labels = data
					images, labels = images.to(self.device), labels.to(self.device)
					outputs=[]
					for i in range(self.ensemble_num):
					  outputs.append(self.model[i](images))
					  outputs[i]=outputs[i].cpu().numpy()

					labels = labels.cpu().numpy()
					for i in range(len(outputs[0])):
						d = np.concatenate((outputs[0][i], outputs[1][i]), axis=None)
						d = np.concatenate((d, outputs[2][i]), axis=None)
						d = np.concatenate((d, outputs[3][i]), axis=None)
						d = np.concatenate((d, outputs[4][i]), axis=None)
						d = np.concatenate((d, outputs[5][i]), axis=None)
						d = np.concatenate((d, outputs[6][i]), axis=None)
					 
						self.encoded_train_data.append(d)
						self.encoded_train_label.append(labels[i])

		X_train, self.X_test, y_train, self.y_test = train_test_split(self.encoded_train_data, self.encoded_train_label, test_size=0.05, random_state=0)

		self.mlp = MLPClassifier(alpha=1, max_iter=1000)
		self.mlp.fit(X_train, y_train)
	def show_test_result(self):
		pred = self.mlp.predict(self.X_test)
		total = 0
		correct = 1
		for i in range(len(self.y_test)):
			total += 1
			if self.y_test[i] == pred[i]:
				correct += 1
		print("The correction rate under MLPClassifier is:")
		print(correct * 1.0 / total * 1.0)
