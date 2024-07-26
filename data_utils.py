from os import listdir
from os.path import join
import os
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

class ImageDataset3D(Dataset):
    def __init__(self, folder_path1, folder_path2, common_directories, transform):
        self.transform = transform
        self.sequences = []
        
        for dir_name in sorted(list(common_directories)):
            path1 = os.path.join(folder_path1, dir_name)
            path2 = os.path.join(folder_path2, dir_name)
            directories1 = [d for d in os.listdir(path1) if os.path.isfile(os.path.join(path1, d))]
            directories2 = [d for d in os.listdir(path2) if os.path.isfile(os.path.join(path2, d))]
            common_images = set(directories1).intersection(directories2)     
            lr_list = []
            hr_list = []   
            for image_name in sorted(list(common_images)):
                lr_list.append(os.path.join(path2, image_name))
                hr_list.append(os.path.join(path1, image_name))
            for i in range(len(hr_list)-4):
                self.sequences.append(
                    {'lr': lr_list[i:i+5],
                    'hr': hr_list[i+1:i+4]
                    }
                )

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        path_dict = self.sequences[idx]

        img_hr = torch.stack([self.transform(Image.open(path)) for path in path_dict['hr']])
        img_lr = torch.stack([self.transform(Image.open(path)) for path in path_dict['lr']])

        return img_lr, img_hr


class TestDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.transform = transform
        self.images = []
        directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        for dir_name in sorted(list(directories)):
            path = os.path.join(folder_path, dir_name)
            files = [d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]   
            sequence = []
            for image_name in sorted(list(files)):
                sequence.append(os.path.join(path, image_name))
            
            for i in range(len(sequence)-2):
                self.images.append(
                    sequence[i:i+3]
                )

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        paths = self.images[idx]
        images = torch.stack([self.transform(Image.open(path)) for path in paths])
        return images, paths


class ImportDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.transform = transform
        self.images = []
        directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        for dir_name in list(directories):
            path = os.path.join(folder_path, dir_name)
            files = [d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]   
            for image_name in sorted(list(files)):
                self.images.append(os.path.join(path, image_name))

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path = self.images[idx]
        img = self.transform(Image.open(path))
        return img, path