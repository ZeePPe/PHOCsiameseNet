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
import os
import pickle
import config as C

"""Mario"""

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


dataloader = DataLoader(train_dataset, collate_fn=my_collate)

dataiter = iter(dataloader)


phoc_model = torch.load(C.USED_MODEl, map_location=torch.device('cpu'))

if cuda_id is not None:
    if len(cuda_id) > 1:
        model = nn.DataParallel(phoc_model, device_ids=cuda_id)
        model.cuda()
    else:
        phoc_model.cuda(cuda_id[0])

measurer = SiameseMeasurer(phoc_model)


embeddings = {}

for img, label, path in dataiter:
    fileName = os.path.basename(path[0])
    print(fileName)
    phoc_rep_t = measurer.get_embedding(img[0], sigmoid=False)
    phoc_rep_n = phoc_rep_t.cpu().detach().numpy()
    embeddings[fileName] = phoc_rep_n


with open(C.EMBEDDING_FILE, "wb") as embeddings_file:
    pickle.dump(embeddings, embeddings_file)
