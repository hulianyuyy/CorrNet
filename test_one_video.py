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
from decord import VideoReader, cpu
import argparse
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

def is_image_by_extension(file_path):
    _, file_extension = os.path.splitext(file_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    return file_extension.lower() in image_extensions

def load_video(video_path, max_frames_num=360):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    elif type(video_path) == list:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    else:
        raise ValueError(f"Not support video input : {type(video_path)}")
    total_frame_num = len(vr)
    if total_frame_num> max_frames_num:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return [cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB) for tmp in spare_frames]  # (frames, height, width, channels)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="The path to pretrained weights")
parser.add_argument("--video_path", type=str, help="The path to a video file or a dir contains extracted images from a video")
parser.add_argument("--device", type=int, default=0, help="Which device to run inference")
parser.add_argument("--language", type=str, default='phoenix', choices=['phoenix', 'csl'], help="The target sign language")
parser.add_argument("--max_frames_num", type=int, default=360, help="The max input frames sampled from an input video")
    
args = parser.parse_args()


device_id = args.device  # specify which gpu to use
if args.language == 'phoenix':
    dataset = 'phoenix2014' 
elif args.language == 'csl':
    dataset = 'CSL-Daily' 
else:
    raise ValueError("Please select target language from ['phoenix', 'csl'] in your command")


# Load data and apply transformation
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
gloss_dict = np.load(dict_path, allow_pickle=True).item()

if os.path.isdir(args.video_path): # extracted images of a video
    img_list = []
    for img_path in sorted(os.listdir(args.video_path)):
        cur_path = os.path.join(args.video_path, img_path)
        if is_image_by_extension(cur_path):
            img_list.append(cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB))
elif os.path.splitext(args.video_path)[-1] in VIDEO_FORMATS: # Video case
    try:
        img_list = load_video(args.video_path, args.max_frames_num)  # frames [height, width, channels]
    except Exception as e:
        raise ValueError(f"Error {e} in loading video")
    
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
model = SLRModel( num_classes=len(gloss_dict)+1, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
state_dict = torch.load(args.model_path)['model_state_dict']
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
