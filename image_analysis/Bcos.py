import os
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import NormedConv2d,BcosConv2d,NormedLinear
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# In[3]:


if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available():
	device = torch.device("mps")
else:
	device = torch.device("cpu")


# In[4]:


root_path = './images/'
sup_sets = os.listdir(root_path)
sup_sets = [i for i in sup_sets if (i.endswith('(hom)') or i.endswith('(control)'))] # the two labels
hom = [i for i in sup_sets if i.endswith('(hom)')]
control = [i for i in sup_sets if i.endswith('(control)')]

print("Directories and files in root path:", sup_sets)


# In[5]:


def get_image_path_dir(dir_path):
    recordings = os.listdir(dir_path)
    recordings = [i for i in recordings if not i=='.DS_Store']
    Rs = {key: [] for key in recordings}
    for i in recordings:
        image_names = os.listdir(dir_path+i)
        image_names = [j for j in image_names if j.endswith('.tif')]
        ind = [int(i.split('_')[-1].split('.')[0][1:]) for i in image_names]
        sorted_image_names = [x for _, x in sorted(zip(ind, image_names))]
        Rs[i].append(sorted_image_names)
    return Rs


# In[6]:


control_images = get_image_path_dir(root_path+'/'+control[0]+'/')
hom_images = get_image_path_dir(root_path+'/'+hom[0]+'/')


# In[7]:


paths = []
labels = []
seq = []


for i in hom_images.keys():
    counter = 0
    for j in hom_images[i][0]:
        paths.append(root_path+sup_sets[0]+'/'+i+'/'+j)
        labels.append([0,1])
        seq.append(np.array(counter))
        counter+=1
        if counter>1500:
            break

for i in control_images.keys():
    counter = 0
    for j in control_images[i][0]:
        paths.append(root_path+sup_sets[1]+'/'+i+'/'+j)
        labels.append([1,0])
        seq.append(np.array(counter))
        counter+=1
        if counter>1500:
            break


# In[8]:


class BaseDataset(Dataset):
    def __init__(self, paths, labels, seq, device, transform_data, is_train=True, transform = True):
        train_paths, test_paths, train_labels, test_labels, train_seq, test_seq = train_test_split(
            paths, labels, seq, test_size=0.2, random_state=42)
        self.device = device
        self.transform = transform
        self.transform_data = transform_data
        self.is_train = is_train

        if self.is_train:
            self.paths, self.labels, self.seq = train_paths, train_labels, train_seq
        else:
            self.paths, self.labels, self.seq = test_paths, test_labels, test_seq


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        x = image
        if self.transform:
            x = self.transform_data(x)
        x = x.to(self.device)
        seq_num = torch.tensor(self.seq[index], dtype=torch.long).to(self.device)
        target = torch.tensor(self.labels[index], dtype=torch.float).to(self.device)

        return x, target, seq_num




model = torch.hub.load('B-cos/B-cos-v2', 'vitc_l_patch1_14', pretrained=True)
model[0].linear_head.linear.linear = NormedLinear(1024,2)
model = model.to(device)




train_ds = BaseDataset(paths=paths, labels=labels, seq=seq, device=device, transform_data = model.transform, is_train=True)
test_ds =  BaseDataset(paths=paths, labels=labels, seq=seq, device=device, transform_data = model.transform, is_train=False)



train_loader = DataLoader(train_ds, batch_size= 16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size= 8, shuffle=False)




model.train()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
sigmoid = nn.Sigmoid()




def calculate_accuracy(model, data_loader=test_loader):
    model.eval()
    correct = 0
    total = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for inputs, labels,SQ in tqdm(data_loader):
            outputs = model(inputs.to(device))
            outputs = sigmoid(outputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()
    accuracy = correct / total * 100.0
    return accuracy




epochs = 500
log = []
test_log = 0



counter1 = 0
for epoch in range(epochs):
    counter1 +=1
    model.train()
    running_loss = 0.0
    for inputs, labels,SQ in tqdm(train_loader, desc="Training"):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        outputs = outputs.to(device) #sigmoid(outputs.to(device))
        loss = criterion(outputs.to(device), labels)
        loss.backward()
        optimizer.step()
    test_acc = calculate_accuracy(model,test_loader)
    print (test_acc)
    if test_acc>test_log:
        counter1 = 0
        test_log = test_acc
        torch.save(model.state_dict(), './'+'_model.pth')
    if counter1 > 40:
        break


