import os
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
from torch.utils.data import Dataset



class MyImagesDS(Dataset):

    def __init__(self, root_dir, min_image_width_height=30, transform=None, img_channels=1):
        self.root_dir = root_dir
        self.transform = transform
        self.list_of_images = os.listdir(self.root_dir)
        self.image_iterator = iter(self.list_of_images)
        self.min_image_width_height = min_image_width_height
        self.img_channels = img_channels

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.list_of_images[idx]
        img_path = os.path.join(self.root_dir,img_name)
        image = io.imread(img_path)

         # scale black pixels to 1 and white pixels to 0
        image = 1 - image.astype(np.float32) / 255.0
        
        image = check_size(img=image, min_image_width_height=self.min_image_width_height)

        label = img_name.split("_")[-1].split(".")[0]
       
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def check_size(img, min_image_width_height, fixed_image_size=None):
    '''
    checks if the image accords to the minimum and maximum size requirements
    or fixed image size and resizes if not
    
    :param img: the image to be checked
    :param min_image_width_height: the minimum image size
    :param fixed_image_size:
    '''
    if fixed_image_size is not None:
        if len(fixed_image_size) != 2:
            raise ValueError('The requested fixed image size is invalid!')
        new_img = resize(image=img, output_shape=fixed_image_size[::-1])
        new_img = new_img.astype(np.float32)
        return new_img
    elif np.amin(img.shape[:2]) < min_image_width_height:
        if np.amin(img.shape[:2]) == 0:
            return None
        scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
        new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
        new_img = resize(image=img, output_shape=new_shape)
        new_img = new_img.astype(np.float32)
        return new_img
    else:
        return img


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    path = [item[-1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target, path]