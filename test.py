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

def Test_fusion_resize(img_test1,img_test2,gamma: float,device):
    '''
    img_test1: vi
    img_test2: ir
    gamma:
    '''
    dir = '/data/zsl/Experiment5/trained_model'
    # trained_sh_encoder_VI = torch.load(dir + '/sh_encoder_VI.pth')
    # trained_sh_encoder_IR = torch.load(dir + '/sh_encoder_IR.pth')
    trained_encoder = torch.load(dir + '/encoder.pth')
    trained_decoder = torch.load(dir + '/decoder.pth')

    # img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test1 = torch.from_numpy(img_test1.reshape((1, 3, img_test1.shape[0], img_test1.shape[1])))
    #
    # img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test2 = torch.from_numpy(img_test2.reshape((1, 3, img_test2.shape[0], img_test2.shape[1])))
    x = img_test1.size[0]
    y = img_test1.size[1]

    img_test1 = transform_resize(img_test1)
    img_test1 = img_test1.unsqueeze(0)

    img_test2 = transform_resize(img_test2)
    img_test2 = img_test2.unsqueeze(0)

    img_test1 = img_test1.to(device)
    img_test2 = img_test2.to(device)


    with torch.no_grad():
        sh_M_VI, sh_M_IR,  ex_M_VI, ex_M_IR, _, _,_, _, = trained_encoder(img_test1, img_test2)

    F_sh_M = (sh_M_IR + sh_M_VI) * 0.5
    if gamma == -1:
        F_ex_M = torch.max(ex_M_IR, ex_M_VI)
        # ex_M = torch.max(extra_feature_map_ir, extra_feature_map_vi)
        print('max')
    else:
        F_ex_M = gamma * ex_M_IR + (1 - gamma) * ex_M_VI
        # ex_M = gamma * extra_feature_map_ir + (1 - gamma) * extra_feature_map_vi
        # ex_M = 0.5 * extra_feature_map_ir + 0.5 * extra_feature_map_vi
    with torch.no_grad():
        out = trained_decoder(F_sh_M, F_ex_M)
    return output_img(Resize_of_images(out, x, y))

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

    img_test1, pad = pad_to(img_test1, 32)
    img_test2 = pad_to(img_test2, 32)[0]

    img_test1 = img_test1.to(device)
    img_test2 = img_test2.to(device)


    with torch.no_grad():
        sh_M_VI, sh_M_IR,  ex_M_VI, ex_M_IR, _, _, _, _ = trained_encoder(img_test1, img_test2)

    F_sh_M = (sh_M_IR + sh_M_VI) * 0.5
    if gamma == -1:
        F_ex_M = torch.max(ex_M_IR, ex_M_VI)
        # ex_M = torch.max(extra_feature_map_ir, extra_feature_map_vi)
        print('max')
    else:
        F_ex_M = gamma * ex_M_IR + (0.8 - gamma) * ex_M_VI
        # ex_M = gamma * extra_feature_map_ir + (1 - gamma) * extra_feature_map_vi
        # ex_M = 0.5 * extra_feature_map_ir + 0.5 * extra_feature_map_vi
    with torch.no_grad():
        out = trained_decoder(F_sh_M, F_ex_M)
        out = unpad(out, pad)
    return output_img(out)

def Test_fusion_cat(img_test1,img_test2,gamma: float, device):
    dir = '/data/zsl/Experiment4/trained_model'
    # trained_sh_encoder_VI = torch.load(dir + '/sh_encoder_VI.pth')
    # trained_sh_encoder_IR = torch.load(dir + '/sh_encoder_IR.pth')
    trained_sh_encoder = torch.load(dir + '/sh_encoder.pth')
    trained_ex_encoder_VI = torch.load(dir + '/ex_encoder_VI.pth')
    trained_ex_encoder_IR = torch.load(dir + '/ex_encoder_IR.pth')
    trained_generator = torch.load(dir + '/generator.pth')

    # img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test1 = torch.from_numpy(img_test1.reshape((1, 3, img_test1.shape[0], img_test1.shape[1])))
    #
    # img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
    # img_test2 = torch.from_numpy(img_test2.reshape((1, 3, img_test2.shape[0], img_test2.shape[1])))

    img_test1 = transform(img_test1)
    img_test1 = img_test1.unsqueeze(0)

    img_test2 = transform(img_test2)
    img_test2 = img_test2.unsqueeze(0)

    img_test1, pad = pad_to(img_test1, 4)
    img_test2 = pad_to(img_test2, 4)[0]

    img_test1 = img_test1.to(device)
    img_test2 = img_test2.to(device)

    with torch.no_grad():
        # sh_M_VI = trained_sh_encoder_VI(img_test1)
        # sh_M_IR = trained_sh_encoder_IR(img_test2)
        out_sh = trained_sh_encoder(img_test1, img_test2)
        sh_M_VI = out_sh.feature_map_vi
        sh_M_IR = out_sh.feature_map_ir
        out_ex_vi = trained_ex_encoder_VI(img_test1)
        ex_M_VI = out_ex_vi.exclusive_feature_map
        out_ex_ir = trained_ex_encoder_IR(img_test2)
        ex_M_IR = out_ex_ir.exclusive_feature_map


    F_sh_M = (sh_M_IR + sh_M_VI) * 10000
    # c1 = (c1_ir + c1_vi) * 0.5
    # c2 = (c2_ir + c2_vi) * 0.5
    # F_M = (e_M_VI + e_M_IR) * 0.5
    # F_M = torch.max(e_M_VI, e_M_IR)
    if gamma == -1:
        F_ex_M = torch.max(ex_M_IR, ex_M_VI)
        print('max')
    else:
        F_ex_M = gamma * ex_M_IR + (1 - gamma) * ex_M_VI

    with torch.no_grad():
        out = trained_generator(F_sh_M, F_ex_M)
        out = unpad(out, pad)


    return out



if __name__ == "__main__":
    device = 'cuda:6'
    test_data_path = '/data/zsl/MIRFuse/dataset/TNO_Test'
    # test_data_path = '/data/zsl/MIRFuse/dataset/RS'
    c = 628000
    # Determine the number of files
    Test_Image_Number = len(os.listdir(test_data_path + '/ir'))
    for i in range(int(Test_Image_Number)):
        if c == 10:
            if i == 17 or i == 18 or i == 39 or i == 9 or i == 22:
                Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
                Test_Vis = transform(Test_Vis)
                Test_Vis = Test_Vis.unsqueeze(0)
                out_cat = Test_Vis
                out_cat = out_cat.to(device)
                for gamma in np.arange(0, 1.2, 0.2):
                    # Test_IR = Image.open(test_data_path + '/IR' + str(i + 1) + '.bmp')  # infrared image
                    # Test_Vis = Image.open(test_data_path + '/VIS' + str(i + 1) + '.bmp')  # visible image
                    Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
                    Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
                    out = Test_fusion_cat(Test_Vis, Test_IR, gamma, device)
                    out_cat = torch.cat([out_cat, out], dim=2)

                Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
                Test_IR = transform(Test_IR)
                Test_IR = Test_IR.unsqueeze(0)
                Test_IR = Test_IR.to(device)
                out_cat = torch.cat([out_cat, Test_IR], dim=2)
                img = output_img(out_cat)



                # imsave('/data/zsl/SDIM/Test_result/FLIR_result/F' + str(i + 1 ) + 'gamma=' + str(gamma) +  '.png', Fusion_image)
                imsave('/data/zsl/Experiment4/test_result/cat/c' + str(i + 1)  + '.png', img)

            else:
                pass
        elif c == 100 :
            if i < 9:
                Test_IR = Image.open(test_data_path + '/ir/0' + str(i + 1) + '.png')  # infrared image
                Test_Vis = Image.open(test_data_path + '/vi/0' + str(i + 1) + '.png')  # visible image
                Fusion_image = Test_fusion_resize(Test_Vis, Test_IR, 0.4, device)
                imsave('/data/zsl/MIRFuse/test_result/0' + str(i + 1) + '.png', Fusion_image)

            else:
                Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
                Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
                Fusion_image = Test_fusion_resize(Test_Vis, Test_IR, 0.4, device)
                imsave('/data/zsl/MIRFuse/test_result/' + str(i + 1) + '.png', Fusion_image)
        elif c == 56:
            if i < 9:
                pair = ImagePair(test_data_path+'/ir/', test_data_path+'/ir/')
                Test_IR, Test_VI = pair.ir_t, pair.vi_t
                Fusion_image = Test_fusion(Test_VI, Test_IR, 0.3, device)
                pair.save_fus('/data/zsl/MIRFuse/test_result/0' + str(i + 1) + '.jpg',Fusion_image,color=True )

            else:
                pair = ImagePair(test_data_path + '/ir/', test_data_path + '/ir/')
                Test_IR, Test_VI = pair.ir_t, pair.vi_t
                Fusion_image = Test_fusion(Test_VI, Test_IR, 0.3, device)
                pair.save_fus('/data/zsl/MIRFuse/test_result/' + str(i + 1) + '.jpg', Fusion_image, color=True)


        elif c ==628:
            if i == 39:
                Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
                Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
                for gamma in np.arange(0.1, 0.9, 0.15):
                    Fusion_image = Test_fusion(Test_Vis, Test_IR, gamma, device)
                    imsave('/data/zsl/MIRFuse/test_result/' + str(i + 1 + gamma) + '.png', Fusion_image)


            # else:
            #     Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.jpg')  # infrared image
            #     Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.jpg')  # visible image
            #     Fusion_image = Test_fusion(Test_Vis, Test_IR, 0.4, device)
            #     imsave('/data/zsl/MIRFuse/test_result/' + str(i + 1) + '.jpg', Fusion_image)

        else:
            if i < 9:
                Test_IR = Image.open(test_data_path + '/ir/0' + str(i + 1) + '.png')  # infrared image
                Test_Vis = Image.open(test_data_path + '/vi/0' + str(i + 1) + '.png')  # visible image
                Fusion_image = Test_fusion(Test_Vis, Test_IR, 0.3, device)
                imsave('/data/zsl/MIRFuse/test_result/0' + str(i + 1) + '.png', Fusion_image)

            else:
                Test_IR = Image.open(test_data_path + '/ir/' + str(i + 1) + '.png')  # infrared image
                Test_Vis = Image.open(test_data_path + '/vi/' + str(i + 1) + '.png')  # visible image
                Fusion_image = Test_fusion(Test_Vis, Test_IR, 0.3, device)
                imsave('/data/zsl/MIRFuse/test_result/' + str(i + 1) + '.png', Fusion_image)











