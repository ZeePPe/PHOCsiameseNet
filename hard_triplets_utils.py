import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
from torchvision import transforms
from dataset import myTransforms
from dataset.myImageFolder import ImageFolderWithPaths
import pickle
import config as C
from models.utils import *
import numpy as np
from torch.utils.data.sampler import BatchSampler
import textdistance

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
The selector chooses all possible combination within the input subset for the anchor-positive couples.
For each class, is then calculated a set of neirest negative classes (using a function hard_negatives_fn) from which the negative 
example is randomly selected
"""
class HardTripletSelector(TripletSelector):

    def __init__(self, batch, hard_negatives_fn, class_to_idx):
        self.batch = batch
        self.hard_negatives = hard_negatives_fn
        self.class_to_idx = class_to_idx

    def get_triplets(self):
        triplets = []

        for i in range(len(self.batch)):

            if len(self.batch[i]) < 2:
                continue

            anchor_positives = list(combinations(self.batch[i], 2))
            anchor_positives = np.array(anchor_positives)
            hard_classes = self.hard_negatives(i, self.class_to_idx)

            for anchor_positive in anchor_positives:
                hard_class = np.random.choice(hard_classes)
                hard_negative = np.random.choice(self.batch[hard_class])
                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

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
def separate_train_dataset(dataset, n_samples, embeddings):
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

        # farthest_n_samples parametro?
        train_indices = [ind[0] + i for i in farthest_n_samples(imgs, n_samples, embeddings)]
        train.append(train_indices)

        val_indices = [i for i in ind if i not in train_indices][:(2*n_samples)]
        val.append(val_indices)



    return train, val


#### ESEMPIO DI ESECUZIONE PER CREARE LE TRIPLETTE A PARTIRE DAL DATASET ####
if __name__ == "__main__":
    # we open the embedding file and save it in the embedding variable
    with open(C.EMBEDDING_FILE, "rb") as embeddings_file:
        embeddings = pickle.load(embeddings_file)


    train_dataset = ImageFolderWithPaths(root='data/alphabet',
                                    transform=transforms.Compose([
                                                myTransforms.toRGB(),
                                                myTransforms.Resize(),
                                                myTransforms.ToTensor(),
                                            ]))

    class_to_idx = train_dataset.class_to_idx
    # maximum number of samples we want to use in the train subset
    n_samples = 8
    # split the dataset in train_subset and validation_subset
    train_subset, validation_subset = separate_train_dataset(train_dataset, n_samples, embeddings)

    # count how many classes are not empty in the validation_subset
    print(len([pos for pos in validation_subset if len(pos)>0]))

    # then we create the triplets
    triplet_selector = HardTripletSelector(train_subset, edit_distance_hard_negative, class_to_idx)
    triplets = triplet_selector.get_triplets()

    print("done")
