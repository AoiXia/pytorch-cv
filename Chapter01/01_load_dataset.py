from torchvision import datasets, utils, transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def show(img):
    npimg = img.numpy()
    # convert to H*W*C shape
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    cv2.imshow('Display', npimg_tr)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()

# Data transformation
# Define transformations
data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomVerticalFlip(p=1),
                                     transforms.ToTensor()])
root = os.getcwd()
path2data = os.path.join(root, 'data')

# train_data = datasets.MNIST(path2data, train=True, download=True)
train_data = datasets.MNIST(path2data, train=True, download=True, transform=data_transform)
valid_data = datasets.MNIST(path2data, train=False, download=True)

x_train, y_train = train_data.data, train_data.targets
x_val, y_val = valid_data.data, valid_data.targets

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

# add a dimension to tensor to become B*C*H*W
if len(x_train.shape)==3:
    x_train = x_train.unsqueeze(1)
print(x_train.shape)

if len(x_val.shape)==3:
    x_val = x_val.unsqueeze(1)
print(x_val.shape)

x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)
print(x_grid.shape)
show(x_grid)

data_transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomVerticalFlip(p=1),
                                      transforms.ToTensor()])
# Get a sample image from training dataset
img = train_data[0][0]
# Transform sample image
img_tr = data_transform2(img)
img_tr_np = img_tr.numpy()
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(img_tr_np[0], cmap='gray')
plt.title('transformed')
plt.show()


