# 给融合后的黑白图像上色
import os
import kornia
from matplotlib import Path
from image_pair import ImagePair
import cv2

data_set = "RS"
source_path = "/data/zsl/MIRFuse/dataset/"
visit_path = source_path + data_set
ir_path = visit_path + "/" + "ir"
vi_path = visit_path + "/" + "vi"
from_path ="/data/zsl/MIRFuse/SwinFuse_RoadScene/"
to_path = "/data/zsl/MIRFuse/color_image_Swin/" + data_set
if not os.path.exists(to_path):
    os.mkdir(to_path)

for file_name in os.listdir(ir_path):
    ir_image_path = ir_path + "/" + file_name
    vi_image_path = vi_path + "/" + file_name
    pair = ImagePair(ir_image_path, vi_image_path)
    o_image_path = from_path + file_name
    # print(o_image_path)
    o_image = cv2.imread(str(o_image_path), cv2.IMREAD_GRAYSCALE)
    pair.save_fus(Path(to_path + "/" + file_name),kornia.utils.image_to_tensor(o_image),color=True)
