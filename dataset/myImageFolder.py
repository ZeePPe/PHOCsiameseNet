import torch
from torchvision import datasets
from skimage import io
import numpy as np
from skimage.transform import resize
from torch.utils.data.sampler import BatchSampler

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths and read the image as Np Array. 
    Extends torchvision.datasets.ImageFolder
    """

    def __init__(self, root, transform=None, target_transform=None, min_image_width_height=26):
        super(ImageFolderWithPaths, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.min_image_width_height = min_image_width_height

    # override the __getitem__ method. 
    def __getitem__(self, index):
    
        path, target = self.samples[index]    

        sample = io.imread(path)

        # scale black pixels to 1 and white pixels to 0
        sample = 1 - sample.astype(np.float32) / 255.

        sample = check_size(img=sample, min_image_width_height=self.min_image_width_height)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # make a new tuple that includes original and the path
        #tuple_with_path = (original_tuple + (path,))

        return sample, target, path


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


class HardPositiveBatchSampler(BatchSampler):
    """
    BatchSampler samples n_classes (all classes) and within these classes samples the Hardest n_samples. (n_samples min 2)
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class HardtripleteBatchSampler(BatchSampler):
    """
    Bisogna crere un batch che contiene tutte le classi del set
    e inoltre contiene per ogni classe i top n_samples, dove per top si intende gli item più vicini.
    """

    def __init__(self, labels, n_classes, n_samples, batch_list):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size