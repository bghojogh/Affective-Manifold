import time
import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import json

with open('config.json') as f:
    config = json.load(f)

PATH = "./dataset/"

torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

embedding_dims = 2
batch_size = 32
epochs = 1

class MNIST(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()
        
        if self.is_train:            
            self.images = df.iloc[:, 1:].values.astype(np.uint8)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.values.astype(np.uint8)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(28, 28, 1)
        
        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(28, 28, 1)
            
            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(28, 28, 1)
            negative_label = self.labels[negative_item]
            
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))
            
            return anchor_img, positive_img, negative_img, anchor_label, negative_label
        
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor_output: torch.Tensor, positive_output: torch.Tensor, negative_output: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor_output, positive_output)
        distance_negative = self.calc_euclidean(anchor_output, negative_output)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletLossMultipleMargins(nn.Module):
    def __init__(self, margin_matrix):
        super(TripletLossMultipleMargins, self).__init__()
        self.margin_matrix = margin_matrix
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor_output: torch.Tensor, positive_output: torch.Tensor, negative_output: torch.Tensor,
                anchor_label, negative_label) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor_output, positive_output)
        distance_negative = self.calc_euclidean(anchor_output, negative_output)
        batch_size = anchor_output.shape[0]
        term_before_clipping_with_zero = torch.empty(batch_size,)
        for sample_index in range(batch_size):
            margin = self.margin_matrix[anchor_label[sample_index], negative_label[sample_index]]
            term_before_clipping_with_zero[sample_index] = distance_positive[sample_index] - distance_negative[sample_index] + margin
        losses = torch.relu(term_before_clipping_with_zero)
        return losses.mean()

class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

def load_dataset():
    train_df = pd.read_csv(PATH+"train.csv")
    test_df = pd.read_csv(PATH+"test.csv")

    train_ds = MNIST(train_df, 
                    train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    test_ds = MNIST(test_df, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def main():
    train_loader, test_loader = load_dataset()

    model = Network(embedding_dims)
    model.apply(init_weights)
    model = torch.jit.script(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if config['method'] == 'single_margin':
        criterion = torch.jit.script(TripletLoss())
    elif config['method'] == 'multiple_margins':
        margin_matrix = np.asarray(config['margin_matrix'])
        criterion = torch.jit.script(TripletLossMultipleMargins(margin_matrix=torch.from_numpy(margin_matrix)))

    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label, negative_label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            
            if config['method'] == 'single_margin':
                loss = criterion(anchor_out, positive_out, negative_out)
            elif config['method'] == 'multiple_margins':
                loss = criterion(anchor_out, positive_out, negative_out, anchor_label, negative_label)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

    # save the model:
    if not os.path.exists("./saved_files/"):
        os.makedirs("./saved_files/")
    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
            }, "./saved_files/trained_model.pth")

    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, _, _, label, _ in tqdm(train_loader):
            train_results.append(model(img.to(device)).cpu().numpy())
            labels.append(label)
            
    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels==label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()