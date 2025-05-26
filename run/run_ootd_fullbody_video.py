from pathlib import Path
import sys

import numpy as np
from PIL import Image
from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
from util.image_warp import crop2_43

import argparse

parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
#parser.add_argument('--model_path', type=str, default="", required=True)
#parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="dc", required=False)
parser.add_argument('--category', '-c', type=int, default=2, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=1, required=False)
parser.add_argument('--seed', type=int, default=0, required=False)
args = parser.parse_args([])

openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type  # "hd" or "dc" dc: fullbody
category = args.category  # 0:upperbody; 1:lowerbody; 2:dress
cloth_path = "./target_garments/first_garment.jpg"
model_path = ""

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed


def build_model():
    if model_type == "hd":
        model = OOTDiffusionHD(args.gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(args.gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")
    return model


class TryOnModel:
    def __init__(self, cloth_path):
        self.model = build_model()
        self.cloth_img = crop2_43(Image.open(cloth_path)).resize((768, 1024))

    def forward(self, frame: np.ndarray):
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        frame_43 = crop2_43(frame)
        model_img = frame_43.resize((768, 1024))
        keypoints = openpose_model(model_img.resize((384, 512)))
        model_parse, _ = parsing_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        #masked_vton_img.save('./images_output/mask.png')

        images = self.model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=self.cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

        result = images[0]
        result = np.array(result)
        return result[:, :, [2, 1, 0]]


from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from util.target_garment_dict import target_garment_dict
from util.video_dict import video_dict
from util.multithread_video_writer import MultithreadVideoWriter


def gen_result(video_path, cloth_path):
    target_dir = './ootd_results'
    os.makedirs(target_dir, exist_ok=True)
    garment_name = os.path.basename(cloth_path).split('.')[0]
    target_dir = os.path.join(target_dir, garment_name)
    os.makedirs(target_dir, exist_ok=True)
    video_loader = MultithreadVideoLoader(video_path)
    video_name = os.path.basename(video_path)


    video_writer = MultithreadVideoWriter(os.path.join(target_dir, video_name),
                                          fps=video_loader.get_fps())
    tryon_model = TryOnModel(cloth_path)
    for i in range(len(video_loader)):
        print(i, '/', len(video_loader))
        #if i > 10:
        #    break
        frame = tryon_model.forward(video_loader.cap())
        video_writer.append(frame)
    video_writer.make_video()
    video_writer.close()


if __name__ == '__main__':
    #video_path = os.path.join('../quantitative_videos/', video_dict[video_id])
    #cloth_path = os.path.join('../fullbody_garments', 'han.jpg')
    gen_result('../raw_videos/jin_16_test.mp4', '../fullbody_garments/han.jpg')
    gen_result('../raw_videos/jin_16_train.mp4', '../fullbody_garments/han.jpg')
    gen_result('../raw_videos/jin_16_test.mp4', '../fullbody_garments/dress.jpg')
    gen_result('../raw_videos/jin_16_train.mp4', '../fullbody_garments/dress.jpg')
    gen_result('../raw_videos/jin_16_test.mp4', '../fullbody_garments/coat.jpg')
    gen_result('../raw_videos/jin_16_train.mp4', '../fullbody_garments/coat.jpg')
    gen_result('../raw_videos/jin_16_test.mp4', '../fullbody_garments/korean.jpg')
    gen_result('../raw_videos/jin_16_train.mp4', '../fullbody_garments/korean.jpg')
