import torch
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision import transforms

from trainer.MIRFuse_trainer import MIRFuse_run

import datetime

transforms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

train_data_path = '/data/zsl/MIRFuse/dataset/M3FD_FiveCrop_128'
dir = '/data/zsl/MIRFuse/trained_model'
plt_dir = '/data/zsl/MIRFuse/plt_result'

img_size = 128
epochs = 40
lr = 1e-4

device_ids = [6,7,3,4]  # 选中几张卡
device_prime = "cuda:6"

def data(batch_size):
    Train_Image_Number = len(os.listdir(train_data_path + '/VI/VIS/'))
    Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size

    root_VIS = train_data_path + '/VI/'
    root_IR = train_data_path + '/IR/'

    Data_VIS = torchvision.datasets.ImageFolder(root_VIS, transform=transforms)
    dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, batch_size, shuffle=False)

    Data_IR = torchvision.datasets.ImageFolder(root_IR, transform=transforms)
    dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size, shuffle=False)

    return Iter_per_epoch, dataloader_VIS, dataloader_IR

if __name__ == "__main__":

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    Iter_per_epoch, dataloader_VIS, dataloader_IR = data(batch_size=32)
    MIRFuse_run(epochs=epochs, lr=lr, dataloader_VIS=dataloader_VIS, dataloader_IR=dataloader_IR,
             Iter_per_epoch=Iter_per_epoch, device_ids=device_ids, device_prime=device_prime,
             dir=dir, plt_dir=plt_dir)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
