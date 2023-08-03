import argparse
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from util.add_watermark import watermark_image

# Define the transformer for image preprocessing
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def parse_args():
    parser = argparse.ArgumentParser(description="Video Face Swapping")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the options and set other face swapping options
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f')  ## dummy arg to avoid bug
    opt = opt.parse()
    opt.pic_a_path = args.image_path
    opt.video_path = args.video_path
    opt.output_path = args.output_path
    opt.temp_path = './tmp'
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    opt.use_mask = False  ## new feature up-to-date
    opt.crop_size = 224

    # Load the pre-trained model
    model = create_model(opt)
    model.eval()

    # Initialize the face detection and alignment model
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # Read the input image and perform face detection and alignment
    with torch.no_grad():
        pic_a = opt.pic_a_path
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole, opt.crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # Convert numpy to tensor
        img_id = img_id.cuda()

        # Create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to('cuda')

    # Perform video face swapping and save the output video
    video_swap(opt.video_path, latend_id, model, app, opt.output_path, temp_results_dir=opt.temp_path, use_mask=opt.use_mask)

if __name__ == "__main__":
    main()
