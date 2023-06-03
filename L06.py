# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
# from IPython.display import clear_output

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])

train_set = MNIST(root='./MNIST', train=True, download=True, transform=transform)
test_set = MNIST(root='./MNIST', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# clear_output()

# %%
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np

x_train = train_set.data.numpy() # images
y_train = train_set.targets.numpy() # labels
labels_names = list(map(str, range(10)))
plt.figure(figsize = (20.0, 20.0))  
for i in range(10):  # for all classes (0 to 9)
  label_indexes = np.where(y_train == i)[0] # get indexes for each class 
  index = np.random.choice(label_indexes)
  img = x_train[index]

  plt.subplot(1, 10, i + 1)  
  plt.title(labels_names[i])  
  plt.axis("off")  
  imshow(img,cmap='gray')

# %%
import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Cuda available: ", torch.cuda.is_available(), '\n')

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to CUDA
print(f"Is CUDA built? {torch.backends.cuda.is_built()}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# device = "cpu"
print(f"Using device: {device}")


# %%
class CNN_model(nn.Module):

    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # in channel=1, out=32
            nn.MaxPool2d(2), # size [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), # in channel=32, out=32
            nn.MaxPool2d(2), # size [32,7,7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*7*7, 100), # in = channel*heght*width
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x): 
        x = self.conv_stack(x) 
        return x


# %%
from timeit import default_timer

model = CNN_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
duration_sum = 0
num_epochs = 10
loss_hist = [] # for plotting

for epoch in range(num_epochs):
    start = default_timer()
    hist_loss = 0
    for _, batch in enumerate(train_loader, 0): # get batch
        # parse batch 
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        # sets the gradients of all optimized tensors to zero.
        optimizer.zero_grad() 
        # get outputs
        y_pred = model(imgs) 
        # calculate loss
        loss = criterion(y_pred, labels)
        # calculate gradients
        loss.backward() 
        # performs a single optimization step (parameter update)
        optimizer.step()
        hist_loss += loss.item()
    loss_hist.append(hist_loss / len(train_loader))
    duration = default_timer() - start
    duration_sum += duration
    print(f"Epoch={epoch} loss={loss_hist[epoch]:.4f} time:{duration} s")
    
print(f'Mean epoch duration: {duration_sum / num_epochs} s')

# %%
plt.plot(range(num_epochs), loss_hist)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.show()


# %%
def calaculate_accuracy(model, data_loader):
    correct, total = 0, 0 
    with torch.no_grad(): 
        for batch in data_loader: # get batch
            imgs, labels = batch # parse batch
            imgs, labels = imgs.to(device), labels.to(device)
            y_pred = model.forward(imgs) # get output
            _, predicted = torch.max(y_pred.data, 1) # get predicted class
            total += labels.size(0) # all examples
            correct += (predicted == labels).sum().item() # correct predictions 
    return correct / total


# %%
acc_train = round(calaculate_accuracy(model, train_loader), 3)
print(f"Accuracy train = {acc_train}")
acc_test = round(calaculate_accuracy(model, test_loader), 3)
print(f"Accuracy test = {acc_test}")

# %%
# get batch
imgs, labels = next(iter(test_loader))
imgs, labels = imgs.to(device), labels.to(device)
print('imgs shape: ', imgs.shape)

# %%
# get output
pred = model(imgs)
print('pred shape: ', pred.shape)

# %%
# First sample in prediction batch
pred[0]

# %%
# Calculate probabilities
nn.Softmax(dim=0)(pred[0].detach())

# %%
# remove axis
imgs = torch.reshape(imgs, (64, 28, 28))
print('imgs shape(after reshape): ', imgs.shape)

# %%
# take 10 first images
imgs = imgs[:10]
print('imgs shape: ', imgs.shape)

# %%
pred = pred[:10].detach()
print('Prediction(1 sample):\n', pred[0])
digits = np.argmax(pred.cpu().numpy(), axis=1)
print('Predicted class: ', digits[0])

# %%
plt.figure(figsize = (25.0, 25.0))
for i in range(10):
  img = imgs[i]

  plt.subplot(1, 10, i + 1)
  plt.title('pred: ' + str(digits[i]) + ' real: '+str(labels[i].cpu().numpy())) # predicted and real values
  plt.axis("off")
  imshow(img.cpu().numpy(),cmap='gray')

# %%

# %%
