import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import myTransforms
from dataset.myDataset import my_collate
from dataset.myImageFolder import ImageFolderWithPaths

from siamese_measurer import SiameseMeasurer
from models.networks import FrozenPHOCnet


BASE_MODEL_NAME = 'weights/PHOC_best.pth'
TRAINED_MODEL_NAME = 'weights/PHOC_best_trained.pth'
MODEL_NAME = TRAINED_MODEL_NAME

cuda = torch.cuda.is_available()

if cuda:
    cuda_id = [0]
else:
    cuda_id = None


train_dataset = ImageFolderWithPaths(root='data/limited_alphabet',
                                transform=transforms.Compose([
                                            myTransforms.toRGB(),
                                            myTransforms.Resize(),
                                            myTransforms.ToTensor(),
                                        ]))


dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=my_collate)
dataiter = iter(dataloader)

img_batch, labels_batch, paths_batch = next(dataiter)

# take the first img from the batch
img1 = img_batch[9]
label1 = labels_batch[9]
path1 = paths_batch[9]

img2 = img_batch[1]
label2 = labels_batch[1]
path2 = paths_batch[1]


img3 = img_batch[8]
label3 = labels_batch[8]
path3 = paths_batch[8]


# Load PHOCnet model
#phoc_model = torch.load(MODEL_NAME)
freeze_phoc_model = FrozenPHOCnet(MODEL_NAME)
#embedding_net = EmbeddingNet()


model = freeze_phoc_model

measurer = SiameseMeasurer(model, cuda_id)

phoc_rep = measurer.get_embedding(img1)

distance12 = measurer.get_distance(img1, img2)
distance13 = measurer.get_distance(img1, img3)

#print(phoc_rep)

print(f"Image 1 -> class{label1.item()}, file:{path1}")
print(f"Image 1 -> class{label2.item()}, file:{path2}")
print(f"Distance = {distance12}")

print(f"Image 1 -> class {label1.item()}, file: {path1}")
print(f"Image 1 -> class {label3.item()}, file: {path3}")
print(f"Distance = {distance13}")



