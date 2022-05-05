import sys
import numpy as np
from models.utils import TripletSelector
import os
import textdistance
import torch


def getMinorDistanceNegative(anchor, neirest_negatives, embeddings):

    min_distance = sys.maxsize

    for (img,l) in neirest_negatives:
        distance = np.linalg.norm(embeddings[os.path.basename(anchor)] - embeddings[os.path.basename(img)])
        if distance < min_distance:
            min_distance = distance
            img_min = img
            label = l

    return img_min, label

def getMinorDistancePair(imgs, embeddings):

    min_distance = sys.maxsize

    for img1 in imgs:
        for img2 in imgs:
            if img1 == img2:
                continue
            distance = np.linalg.norm(embeddings[os.path.basename(img1)]-embeddings[os.path.basename(img2)])
            if distance < min_distance:
                min_distance = distance
                img_min1 = img1
                img_min2 = img2

    return img_min1, img_min2

def getMajorDistancePair(imgs, embeddings):

    max_distance = 0

    for img1 in imgs:
        for img2 in imgs:
            if img1 == img2:
                continue
            distance = np.linalg.norm(embeddings[os.path.basename(img1)]-embeddings[os.path.basename(img2)]) # DISTANZA EUCLIDEA IN NUMPY
            if distance > max_distance:
                max_distance = distance
                img_max1 = img1
                img_max2 = img2

    return img_max1, img_max2

def findNeirestNegatives(ngram, classes, samples, class_to_idx):

    idx = classes.index(ngram)

    classes = [c.replace('_', '') for c in classes]

    ngram = classes[idx]

    edit_distances = {n:textdistance.levenshtein(ngram, n) for n in classes if n.replace(' ', '') != ngram.replace(' ', '')}
    edit_distances = {k: v for k, v in sorted(edit_distances.items(), key=lambda item: item[1])}

    first_ten = [ngram for ngram in list(edit_distances.keys())[:10]]


    neirest_negatives = []
    for n in first_ten:

        neirest_negatives.append([(img,l) for (img, l) in samples if l == classes.index(n)][0])


    return  neirest_negatives

class HardPositiveHardNegativeTripletSelector(TripletSelector):

    def __init__(self):
        super(HardPositiveHardNegativeTripletSelector, self).__init__()


    def get_triplets(self, dataloader, embeddings):

        samples = dataloader.dataset.samples
        classes = dataloader.dataset.classes
        class_to_idx = dataloader.dataset.class_to_idx
        triplets = []
        triplets_indices = []

        for c in set(classes):
            print(c)
            l = class_to_idx[c]
            class_samples = [img for (img, label) in samples if label == l]

            if len(class_samples) == 1:
                continue

            if len(class_samples) == 2:
                anchor, positive = class_samples
            else:
                anchor, positive = getMajorDistancePair(class_samples, embeddings)

            neirest_negatives = findNeirestNegatives(c, classes, samples, class_to_idx)
            negative, nl = getMinorDistanceNegative(anchor, neirest_negatives, embeddings)

            triplets_indices.append([samples.index((anchor,l)), samples.index((positive,l)), samples.index((negative,nl))])
            triplets.append([anchor, positive, negative])

        triplets_indices = np.array(triplets_indices)

        return torch.LongTensor(triplets_indices)
