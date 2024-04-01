from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

#----------------PARAM----------------
writer = SummaryWriter("logs")
train_set = r'DATASET\Training_SET'
validation_set = r'DATASET\Validation_SET'
train_value = False # Training or Eval
train_epoch = 15
lr= 1e-4
batch_size = 16
num_classes = 3
input_size = 128, 128

print('   Sys V: ', sys.version)
print(' Pytorch: ', torch.__version__)
print('   Numpy: ', np.__version__)
print('Pandas V: ', pd.__version__)

#----------------Check for GPU(NVIDIA-CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print('Hardware: ', torch.cuda.get_device_name(0), '|', device)

#----------------DATALOADER----------------
transform = transforms.Compose([
    transforms.Resize((input_size)),
    transforms.ToTensor(),
])

class ImageDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = ImageFolder(main_dir, transform=transform)

    def __len__(self):
        return len(self.main_dir)

    def __getitem__(self, idx):
        return self.main_dir[idx]
    
    @property
    def classes(self):
        return self.data.classes

    def train(self, train_epoch, lr, batch_size, dataloader, validation_loader):
        model = ImageClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        losses, val_losses = [], []

        pbar = tqdm(total=train_epoch, colour="green")

        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(train_epoch):
            model.train()
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
            epoch_loss = running_loss / len(dataloader)
            losses.append(epoch_loss)  
            writer.add_scalar('Loss/train', epoch_loss, epoch)  # Log training loss

            #----------------VALIDATE----------------
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)
                val_loss = running_loss / len(validation_loader.dataset)
                val_losses.append(val_loss)       
                writer.add_scalar('Loss/validation', val_loss, epoch)  # Log validation loss
                #print(f"Epoch {epoch+1}/{train_epoch} - Train loss: {losses}, Validation loss: {val_loss}")
            pbar.update(1)
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 5:
                    break
        writer.add_hparams(
            {'L_R': lr, 'Batch': batch_size},
            {'Losses': sum(losses)/len(losses), 'Validation': sum(val_losses)/len(val_losses)})
        pbar.close()
        writer.close()
        print('Finished Training')

        #----------------PLOT----------------
        plt.figure(figsize=(10, 6))  
        plt.plot(losses, marker='x', linestyle='-', color='b', label='Training Loss')  
        plt.plot(val_losses, marker='x', linestyle='-', color='g', label='Validation Loss')
        plt.title(datetime.now(), fontsize=15) 
        plt.xlabel('Epoch', fontsize=12) 
        plt.ylabel('Loss', fontsize=12)  
        plt.grid(True) 
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4) 
        plt.tight_layout()  
        plt.legend()
        plt.gca().set_facecolor('lightgray')
        plt.show()
        #----------------SAVE----------------
        torch.save(model.state_dict(), 'model.pth')

#----------------CLASSES DICT----------------
target_to_class = {v: k for k, v in ImageFolder(train_set).class_to_idx.items()}
print('CLASS DICT: ', target_to_class)

dataset = ImageDataset(train_set)
print('Number of images dataset: ', len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

validation_dir = ImageDataset(validation_set)
print('Number of images validation: ', len(validation_dir))
validation_loader = DataLoader(validation_dir, batch_size=batch_size, shuffle=True)

#----------------MODEL CLASS----------------
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, (3,3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*(128-6)*(128-6),num_classes)
        )

    def forward(self, x):
        return self.model(x)


#----------------Training loop----------------
if train_value == True:
    #----------------model, loss function and optimizer
    model = ImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses, val_losses = [], []

    pbar = tqdm(total=train_epoch, colour="green")

    for epoch in range(train_epoch):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)  
        writer.add_scalar('Loss/train', epoch_loss, epoch)  # Log training loss
        
        #----------------VALIDATE----------------
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(validation_loader.dataset)
            val_losses.append(val_loss)       
            writer.add_scalar('Loss/validation', val_loss, epoch)  # Log validation loss
            #print(f"Epoch {epoch+1}/{train_epoch} - Train loss: {losses}, Validation loss: {val_loss}")
        pbar.update(1)
    writer.add_hparams(
        {'L_R': lr, 'Batch': batch_size},
        {'Losses': sum(losses)/len(losses), 'Validation': sum(val_losses)/len(val_losses)})
    pbar.close()
    writer.close()
    print('Finished Training')

    #----------------PLOT----------------
    plt.figure(figsize=(10, 6))  
    plt.plot(losses, marker='x', linestyle='-', color='b', label='Training Loss')  
    plt.plot(val_losses, marker='x', linestyle='-', color='g', label='Validation Loss')
    plt.title(datetime.now(), fontsize=15) 
    plt.xlabel('Epoch', fontsize=12) 
    plt.ylabel('Loss', fontsize=12)  
    plt.grid(True) 
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4) 
    plt.tight_layout()  
    plt.legend()
    plt.gca().set_facecolor('lightgray')
    plt.show()
    #----------------SAVE----------------
    torch.save(model.state_dict(), 'model.pth')
else:
    pass
#----------------END OF Training loop----------------
