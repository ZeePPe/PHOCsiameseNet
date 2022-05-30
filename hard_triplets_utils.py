import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from models.utils import *
import numpy as np
from torch.utils.data.sampler import BatchSampler
import textdistance
import cv2

"""
Augment dataset images in such a way each class has at least min_n_samples samples. 
If a class has less than min_n_samples samples, augmentation is performed on the images for creating new samples
"""
# CONTROLLA COME VIENE PASSSATA LA FUNZIONE DI AUGMENTATION, I PARAMETRI NON DOVREBBERO ESSERE SETTATI DENTRO QUESTO METODO! (mode= amount=)
def simple_dataset_augmentation(root, min_n_samples, augmentation_fn):

    folders = [folder for folder in os.listdir(root) if not folder.startswith('.')]
    for folder in folders:
        n = len(os.listdir(root + folder))

        while n < min_n_samples:
            file = np.random.choice([file for file in os.listdir(root + folder) if not file.startswith('.')])
            path = root + folder + '/' + file
            image = cv2.imread(path, 0)
            augment = augmentation_fn(image, mode='s&p', amount = 0.03)
            augment = np.array(augment)
            new_file = file[:8] + str(int(file.split('_')[2]) + 1) + file[9:]
            new_path = root + folder + '/' + new_file
            cv2.imwrite(new_path, 255 * augment)
            n = len(os.listdir(root + folder))

"""
Returns the indices of the first ten classes that have the most similar label to the input class
with respect to the edit distance
"""
def edit_distance_hard_negative(idx, class_to_idx):
    classes = [c.replace('_', '') for c in list(class_to_idx.keys())]
    ngram = classes[idx]

    edit_distances = {n:textdistance.levenshtein(ngram, n) for n in classes if n != ngram}
    edit_distances = {k: v for k, v in sorted(edit_distances.items(), key=lambda item: item[1])}

    first_ten = [classes.index(ngram) for ngram in list(edit_distances.keys())[:10]]

    return first_ten


"""
Selector that chooses all possible combination within the input subset for the anchor-positive couples.
For each class, is then calculated a set of neirest negative classes (using a function hard_negatives_fn) from which the negative 
example is randomly selected 
"""
class HardTripletSelector(TripletSelector):

    def __init__(self, hard_negatives_fn, class_to_idx, p_hardneg=0.8):
        super(HardTripletSelector, self).__init__()
        self.hard_negatives = hard_negatives_fn
        self.class_to_idx = class_to_idx
        self.p_hardneg = p_hardneg
        self.p_otherneg = 1-p_hardneg

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue

            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)
            hard_classes = self.hard_negatives(label, self.class_to_idx)

            for anchor_positive in anchor_positives:

                hard_class_samples = [np.where(labels == hard_class)[0] for hard_class in hard_classes]
                hard_class_samples = [samples for samples in hard_class_samples if len(samples) > 0]
                hard_class_samples = [sample for samples in hard_class_samples for sample in samples]

                negative_indices = np.where(np.logical_not(label_mask))[0]
                negative = np.random.choice(negative_indices)

                if len(hard_class_samples) > 0:
                    hard_negative = np.random.choice(hard_class_samples)
                    negative = np.random.choice([hard_negative, negative], p = [self.p_hardneg, self.p_otherneg])

                triplets.append([anchor_positive[0], anchor_positive[1], negative])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)




def get_farthest_from(result, imgs, embeddings):

    dist = {img:0 for img in imgs if img not in result}
    diff = {img:0 for img in imgs if img not in result}

    for img in dist:
        for res in result:
            new_dist = np.linalg.norm(embeddings[os.path.basename(img)] - embeddings[os.path.basename(res)])
            if dist[img] != 0:
                diff[img] += abs(new_dist - dist[img])
            dist[img] += new_dist

    dist = {k:v for k,v in sorted(dist.items(), key= lambda item:item[1], reverse=True)}

    if len(result) == 1:
        return list(dist.keys())[0]

    diff = {k: v for k,v in sorted(diff.items(), key=lambda item: item[1])}

    if len(dist) == 1:
        return list(dist.keys())[0]

    for i in range(1,len(dist) + 1):
        for img in list(dist.keys())[:i]:
            if img in list(diff.keys())[:i]:
                return img


def farthest_n_samples(imgs, n_samples, embeddings):
    if len(imgs) < 3:
        return [imgs.index(i) for i in imgs]

    result = []
    result.append(np.random.choice(imgs))
    while len(imgs) > len(result) and n_samples > len(result):
        result.append(get_farthest_from(result, imgs, embeddings))

    return [imgs.index(i) for i in result]

"""
The training set is divided in a train_subset in which, for each class, n_samples samples are selected.
The selected ones are as distant as possible from each other, according to a measure of distance expressed by the embedding file.
In this way, it is possible to create hard triplets by randomly choosing in the train_subset.
The validation_subset is made up of the samples that are not used in the train_subset, with a maximum of (2*n_samples) samples per class. 
"""
def separate_train_dataset(dataset, n_samples, embeddings, best_n_samples_fn):
    labels = torch.tensor(dataset.targets, dtype = torch.int64)
    labels_set = list(set(labels.numpy()))
    class_to_idx = dataset.class_to_idx
    samples = dataset.samples
    label_to_indices = {label: np.where(labels.numpy() == label)[0] for label in labels_set}
    classes = np.array(list(class_to_idx.keys()))

    train = []
    val = []

    for class_ in classes:
        label = class_to_idx[class_]
        ind = label_to_indices[label]
        imgs = [samples[i][0] for i in ind]

        train_indices = [ind[0] + i for i in best_n_samples_fn(imgs, n_samples, embeddings)]
        train.append(train_indices)

        val_indices = [i for i in ind if i not in train_indices][:(2*n_samples)]
        val.append(val_indices)

    train = [idx for class_ in train for idx in class_]
    val = [idx for class_ in val for idx in class_]

    return train, val


class HardPositiveBatchSampler(BatchSampler):
    """
    BatchSampler samples n_classes (all classes) and within these classes samples the Hardest n_samples. (n_samples min 2)
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, subset, n_classes, n_samples):
        self.subset = subset
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))

        self.subset_labels = [self.labels.numpy()[ind] for ind in self.subset.indices]
        self.label_to_indices = {label: np.where(self.subset_labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.subset)

        self.batch_size = 0
        for l in self.labels_set:
            if len(self.label_to_indices[l]) < n_samples:
                self.batch_size += len(self.label_to_indices[l])
            else:
                self.batch_size += n_samples


    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) # Replace False da errore
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

