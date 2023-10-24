import torch
import os
from PIL import Image
from skimage.io import imsave
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from image_pair import ImagePair, ImageFuse


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

transform_resize = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor(),
    transforms.Resize([128, 128])
])

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

def Resize_of_images(img, x:int, y:int):
    resize = transforms.Compose([transforms.Resize([y, x])])
    return resize(img)

def Test_fusion(img_test1,img_test2,gamma: float,device):
    '''
    img_test1: vi
    img_test2: ir
    gamma:
    '''
    dir = '/data/zsl/MIRFuse/trained_model'
    # trained_sh_encoder_VI = torch.load(dir + '/sh_encoder_VI.pth')
    # trained_sh_encoder_IR = torch.load(dir + '/sh_encoder_IR.pth')
    trained_encoder = torch.load(dir + '/encoder.pth')
    trained_decoder = torch.load(dir + '/decoder.pth')

    # img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test1 = torch.from_numpy(img_test1.reshape((1, 3, img_test1.shape[0], img_test1.shape[1])))
    #
    # img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test2 = torch.from_numpy(img_test2.reshape((1, 3, img_test2.shape[0], img_test2.shape[1])))

    img_test1 = transform(img_test1)
    img_test1 = img_test1.unsqueeze(0)

    img_test2 = transform(img_test2)
    img_test2 = img_test2.unsqueeze(0)

    # img_test1, pad = pad_to(img_test1, 32)
    # img_test2 = pad_to(img_test2, 32)[0]

    img_test1 = img_test1.to(device)
    img_test2 = img_test2.to(device)


    with torch.no_grad():
        sh_M_VI, sh_M_IR,  ex_M_VI, ex_M_IR, _, _, _, _ = trained_encoder(img_test1, img_test2)

    # F_sh_M = (sh_M_IR + sh_M_VI) * 0.5
    # if gamma == -1:
    #     F_ex_M = torch.max(ex_M_IR, ex_M_VI)
    #     # ex_M = torch.max(extra_feature_map_ir, extra_feature_map_vi)
    #     print('max')
    # else:
    #     F_ex_M = gamma * ex_M_IR + (0.8 - gamma) * ex_M_VI
    #     # ex_M = gamma * extra_feature_map_ir + (1 - gamma) * extra_feature_map_vi
    #     # ex_M = 0.5 * extra_feature_map_ir + 0.5 * extra_feature_map_vi
    # with torch.no_grad():
    #     out = trained_decoder(F_sh_M, F_ex_M)
    #     out = unpad(out, pad)
    # sh_M_VI = sh_M_VI.squeeze(0)
    return ex_M_IR

if __name__ == "__main__":
    device = 'cuda:6'
    test_data_path = '/data/zsl/MIRFuse/dataset/TNO_Test'
    Test_Image_Number = len(os.listdir(test_data_path + '/ir'))
    for i in range(int(Test_Image_Number)):
        # if i < 9:
        #     Test_IR = Image.open(test_data_path + '/ir/0' + str(i + 1) + '.png')  # infrared image
        #     Test_Vis = Image.open(test_data_path + '/vi/0' + str(i + 1) + '.png')  # visible image
        #     Fusion_image = Test_fusion(Test_Vis, Test_IR, 0.3, device)
        #     print(Fusion_image.shape)
        #     # imsave('/data/zsl/MIRFuse/test_result/0' + str(i + 1) + '.png', Fusion_image)

        if i == 16:
            Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
            Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
            Fusion_image = Test_fusion(Test_Vis, Test_IR, 0.3, device)
            for j in range(128):

                Feature = output_img(Fusion_image[:,j,:,:].unsqueeze(0))
                imsave('/data/zsl/MIRFuse/feature/' + str(j + 1) + '.png', Feature)
            # print(Fusion_image.shape)

