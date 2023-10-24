from torchvision import transforms
import torch
import os
from PIL import Image
from skimage.io import imsave

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

dir = '/data/zsl/MIRFuse/dataset/RS'

if __name__ == "__main__":
    i = 213
    ir = 'ir'
    vi = 'vi'
    a = Image.open(dir + '/'+ ir + '/' + str(i + 1) + '.jpg')
    a = transform(a)
    a = a.squeeze(0)
    imsave('/data/zsl/MIRFuse/test_result/grey' + ir + str(i + 1) + '.jpg', a)
    b = Image.open(dir + '/' + vi + '/' + str(i + 1) + '.jpg')
    b = transform(b)
    b = b.squeeze(0)
    imsave('/data/zsl/MIRFuse/test_result/grey' + vi + str(i + 1) + '.jpg', b)
