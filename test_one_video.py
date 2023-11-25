#Ref: https://blog.csdn.net/weixin_41735859/article/details/106474768
import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils

device_id = 0  # specify which gpu to use
dataset = 'phoenix2014'
video_path = 'path_to_your_video/*.jpg'  # please extract the video into images previouly. Fill in the path with image suffix (e.g., jpr or png). 
# Example
#video_path = '/disk2/dataset/german_dataset/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/dev/01April_2010_Thursday_heute_default-1/1/*.png'

model_weights = 'path_to_pretrained_weight.pt'


# Load data and apply transformation
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
gloss_dict = np.load(dict_path, allow_pickle=True).item()
img_list = sorted(glob.glob(video_path))
img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
vid, label = transform(img_list, None, None)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)

left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride 
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

max_len = vid.size(1)
video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)

device = utils.GpuDataParallel()
device.set_device(device_id)
# Define model and load state-dict
model = SLRModel( num_classes=1296, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
state_dict = torch.load(model_weights)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)
model = model.to(device.output_device)
model.cuda()

model.eval()

vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
print('output glosses : {}'.format(ret_dict['recognized_sents']))
# Example 
# output glosses : [[('ICH', 0), ('LUFT', 1), ('WETTER', 2), ('GERADE', 3), ('loc-SUEDWEST', 4), ('TEMPERATUR', 5), ('__PU__', 6), ('KUEHL', 7), ('SUED', 8), ('WARM', 9), ('ICH', 10), ('IX', 11)]]
