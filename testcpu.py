import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import sys
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model_small
import numpy as np
from PIL import Image
import glob
import time
from tqdm import tqdm


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of CLIP-LIT (liang. 2023)')
parser.add_argument('-i', '--input', help='directory of input folder', default='./test/')
parser.add_argument('-o', '--output', help='directory of output folder', default='./output/')
parser.add_argument('-c', '--ckpt', help='test ckpt path', default='./pretrained_models/enhancement_model.pth')

args = parser.parse_args()

# U_net = model_small.UNet_emb_oneBranch_symmetry_noreflect(3,1)
U_net = model_small.UNet_emb_oneBranch_symmetry(3,1)

# state_dict = torch.load(args.ckpt)
state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
	name = k[7:] # remove `module.`
	new_state_dict[name] = v
U_net.load_state_dict(new_state_dict)
# U_net.cuda()
U_net.to(torch.device('cpu'))
#U_net.load_state_dict(torch.load('./pretrained_models/enhancement_model.pth'))

def lowlight(image_path): 

	data_lowlight = Image.open(image_path)#.convert("RGB")
	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	
	# data_lowlight = torch.from_numpy(data_lowlight).float().cuda()
	data_lowlight = torch.from_numpy(data_lowlight).float().to(torch.device('cpu'))
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0) 

	light_map = U_net(data_lowlight)
	enhanced_image = torch.clamp((data_lowlight / light_map), 0, 1)

	image_path = args.output+os.path.basename(image_path)

	image_path = image_path.replace('.jpg','.png')
	image_path = image_path.replace('.JPG','.png')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)
	
if __name__ == '__main__':
    with torch.no_grad():
        filePath = args.input
        file_list = os.listdir(filePath)
        # Sắp xếp danh sách file theo thứ tự tên file
        file_list = sorted(file_list)
        print("Sorted file list:", file_list)
        
        for file_name in tqdm(file_list, desc="Processing images"):
            image = os.path.join(filePath, file_name)
            lowlight(image)

		

