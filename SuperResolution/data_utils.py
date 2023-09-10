import argparse
import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.ARW'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor, interpolation=Image.Resampling.BICUBIC)
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, dataset_name, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir + '/' + dataset_name + '_SRF_' + str(upscale_factor) + '/data'
        self.target_dir = dataset_dir + '/' + dataset_name + '_SRF_' + str(upscale_factor) + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB') 
        target = Image.open(self.target_filenames[index]).convert('RGB')
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_filenames)


def generate_dataset(data_type, upscale_factor):
    dataset_name = 'DIV2K'
    resolution = 256
    images_name = [x for x in listdir('data/' + dataset_name + '/' + data_type) if is_image_file(x)]
    crop_size = calculate_valid_crop_size(resolution, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)

    root = 'data/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/' + dataset_name + '_SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
            + str(upscale_factor) + ' from ' + dataset_name):
        image = Image.open('data/' + dataset_name + '/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)

        image.save(image_path + '/' + image_name)
        target.save(target_path + '/' + image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor

    generate_dataset(data_type='train', upscale_factor=UPSCALE_FACTOR)
    generate_dataset(data_type='val', upscale_factor=UPSCALE_FACTOR)
