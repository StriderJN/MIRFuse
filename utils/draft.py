import os
train_data_path = '/data/zsl/MIRFuse/dataset/M3FD_FiveCrop_128'

Train_Image_Number = len(os.listdir(train_data_path + '/VI/VIS/'))

print(Train_Image_Number)