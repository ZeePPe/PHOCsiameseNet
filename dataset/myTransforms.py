from torchvision.transforms import functional as F
from torchvision import transforms
import torch
import numpy as np
from skimage import transform
from skimage.color import gray2rgb
from PIL import Image


class PHOC_preproces(object):
    def __call__(self, input):
        #y = asd.view(asd.shape[1], asd.shape[2], asd.shape[3])
        #y = torch.squeeze(asd)
        pre_process = transforms.Compose([transforms.ToPILImage(input[1:]),
                                        transforms.Grayscale(num_output_channels=1), 
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.5], std=[0.5]),
                                        ])
        return pre_process


class normalize(object):
    def __call__(self, input):
        image   = np.array(input)
        image = 1 - image.astype(np.float32) / 255.0
       
        return image


class ToGray(object):
    def __call__(self, image, target):
        image = F.Grayscale(image,num_output_channels=1)
        return image, target



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, input):
        #image, label = sample['image'], sample['label']

        #if  len(image.shape) == 4:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            #image = image.transpose((0, 3, 1, 2))
        tensor_image = torch.from_numpy(input)
        return tensor_image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (list): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (list))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if len(self.output_size) == 1:
            scale = float(self.output_size[0]) / float(image.shape[0])
            new_shape = (int(scale * image.shape[0]), int(scale * image.shape[1]))

        if len(self.output_size) == 2:
            new_shape = (self.output_size[0], self.output_size[1])

        img = transform.resize(image=image, output_shape=new_shape).astype(np.float32)

        return {'image': img, 'label': label}

class Resize(object):
    def __call__(self, input):
        
        if len(input.shape) == 2:
            image = input.reshape((1,) + input.shape)
        if len(input.shape) == 3:
            image = input.transpose(2, 0, 1)

        return image

class toRGB(object):
    def __call__(self, input):

        #image = input.convert('RGB')

        image = gray2rgb(input)

        return image