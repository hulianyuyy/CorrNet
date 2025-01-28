import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils
from PIL import Image
import argparse

import numpy as np
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
os.environ['GRADIO_TEMP_DIR'] = 'gradio_temp'
import gradio as gr
import os
import warnings
from decord import VideoReader, cpu
warnings.filterwarnings("ignore")

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

def run_inference(inputs):
    """
    Run inference on one input sample.

    Args:
        args: Command-line arguments.
    """
    img_list = []
    if isinstance(inputs, list):  # Multi-image case
        for x in inputs:
            if is_image_by_extension(x):
                img_list.append(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) )

    elif os.path.splitext(inputs)[-1] in VIDEO_FORMATS: # Video case
        try:
            img_list = load_video(inputs, args.max_frames_num)  # frames [height, width, channels]
        except Exception as e:
            raise ValueError(f"Error {e} in loading video")
    else:
        raise ValueError("Video path is incorrect!")

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

    vid = device.data_to_device(vid)
    vid_lgt = device.data_to_device(video_length)
    ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
    return ret_dict['recognized_sents'] # [[('ICH', 0), ('LUFT', 1), ('WETTER', 2), ('GERADE', 3), ('loc-SUEDWEST', 4), ('TEMPERATUR', 5), ('__PU__', 6), ('KUEHL', 7), ('SUED', 8), ('WARM', 9), ('ICH', 10), ('IX', 11)]]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="The path to pretrained weights")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--language", type=str, default='phoenix', choices=['phoenix', 'csl'])
    parser.add_argument("--max_frames_num", type=int, default=360)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    
    device_id = args.device  # specify which gpu to use
    if args.language == 'phoenix':
        dataset = 'phoenix2014' 
    elif args.language == 'csl':
        dataset = 'CSL-Daily' 
    else:
        raise ValueError("Please select target language from ['phoenix', 'csl'] in your command")

    model_weights = args.model_path

    # Load data and apply transformation
    dict_path = f'./preprocess/{dataset}/gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
    gloss_dict = np.load(dict_path, allow_pickle=True).item()

    device = utils.GpuDataParallel()
    device.set_device(device_id)
    # Define model and load state-dict
    model = SLRModel( num_classes=len(gloss_dict)+1, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
                loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
    state_dict = torch.load(model_weights)['model_state_dict']
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device.output_device)
    model.cuda()

    model.eval()

    def identity(x):
        return x

    with gr.Blocks(title='Continuous sign language recognition') as demo: 
        gr.Markdown("<center><font size=5>Continuous sign language recognition</center></font>")
        gr.Markdown("**Upload multiple images or a video** to get the recognized glossess.")
        with gr.Tab('Multi-Images'):
            with gr.Row():
                with gr.Column(scale=1):
                    multiple_image_show = gr.Gallery(label="Show the input images", height=200)
                    Multi_image_input = gr.UploadButton(label="Click to upload multiple images", file_types = ['.png','.jpg','.jpeg', '.bmp'], file_count = "multiple")
                    multiple_image_button = gr.Button("Run")  
                with gr.Column(scale=1):
                    multiple_image_output = gr.Textbox(label="Output")
        with gr.Tab('Video'):
            with gr.Row():
                with gr.Column(scale=1):
                    Video_input = gr.Video(sources=["upload"], label="Upload a video file")
                    video_button = gr.Button("Run")  
                with gr.Column(scale=1):
                    video_output = gr.Textbox(label="Output")
        multiple_image_button.click(identity, inputs=[Multi_image_input], outputs=multiple_image_show)
        multiple_image_button.click(run_inference, inputs=Multi_image_input, outputs=multiple_image_output)
        video_button.click(run_inference, inputs=Video_input, outputs=video_output)
        
    demo.launch(share=False,server_name="0.0.0.0", server_port=7862)
