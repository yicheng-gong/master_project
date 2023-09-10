import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_image_file
from model_gan import Generator
import cv2
from skimage.metrics import structural_similarity as ssim

def mse(imageA, imageB):
    # Mean Squared Error
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def psnr(imageA, imageB):
    # Peak Signal to Noise Ratio
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

def compute_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    score = ssim(gray1, gray2)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='gan_epoch_4_187.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    dataset_name = 'own'
    path = 'data/test/' + dataset_name + '/SRF_' + str(UPSCALE_FACTOR) + '/test/'
    target_path = 'data/test/' + dataset_name + '/SRF_' + str(UPSCALE_FACTOR) + '/target/'
    images_name = [x for x in listdir(path) if is_image_file(x)]
    model = Generator(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/' + dataset_name + '/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    psnr_list = []
    ssim_list = []
    for image_name in tqdm(images_name, desc='convert LR images to HR images'):
        hr_name = image_name[:-1][:-1][:-1][:-1][:-1][:-1] + 'HR.jpg'
        img = Image.open(path + image_name).convert("RGB")
        target_img = Image.open(target_path + hr_name).convert("RGB")
        target_img = np.array(target_img)
        image = Variable(ToTensor()(img)).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img = out.data[0].numpy()
        out_img = np.transpose(out_img, (1, 2, 0))  # Change to (H, W, 3)
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        psnr_list.append(psnr(np.uint8(out_img), target_img))
        ssim_list.append(compute_ssim(np.uint8(out_img), target_img))
        out_img = Image.fromarray(np.uint8(out_img), 'RGB')
        out_img.save(out_path + hr_name[:-1][:-1][:-1][:-1][:-1][:-1] + 'SR.png')
    
    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    print('Mean PSNR of '+ dataset_name + ' Dataset is: ' + str(mean_psnr) + ' dB')
    print('Mean SSIM of '+ dataset_name + ' Dataset is: ' + str(mean_ssim))
