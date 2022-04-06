import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import os
from sklearn.metrics import SCORERS
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from models.trainer import fit

from dataset import myTransforms
from dataset.myImageFolder import ImageFolderWithPaths
from dataset.myDataset import my_collate

from models.datasets import BalancedBatchSampler

# Set up the network and training parameters
from models.networks import EmbeddingNet
from models.losses import OnlineTripletLoss
from models.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from models.metrics import AverageNonzeroTripletsMetric
from models.networks import FrozenPHOCnet
from train_config import getTrainOptions

#BASE_MODEL_NAME = 'weights/PHOC_best.pth'
SAVE_MODEL_NAME = 'weights/PHOC_trained.pth'
#TRAINING_PATH = 'data/limited_alphabet'
#TEST_PATH = 'data/limited_alphabet'

options = getTrainOptions().parse()

cuda = torch.cuda.is_available()

n_classes = len(os.listdir(options.training_path))
n_samples = options.samples_number
num_workers = options.num_workers
lr = options.learning_rate
n_epochs = options.n_epochs
log_interval = options.log_interval

sch_step = options.sch_step
sch_gamma = options.sch_gamma

frozen_net = True if options.frozen == "True" else False

if cuda:
    cuda_id = [0]
else:
    cuda_id = None
if options.cuda_id[0] == -1:
        cuda_id = None


train_dataset = ImageFolderWithPaths(root=options.training_path,
                                transform=transforms.Compose([
                                            myTransforms.toRGB(),
                                            myTransforms.Resize(),
                                            myTransforms.ToTensor(),
                                        ]))
test_dataset = ImageFolderWithPaths(root=options.test_path,
                                transform=transforms.Compose([
                                            myTransforms.toRGB(),
                                            myTransforms.Resize(),
                                            myTransforms.ToTensor(),
                                        ]))

# split training and test
#test_len = round(per_test*len(train_dataset))
#train_len = len(train_dataset)-test_len
#train_datasett, test_dataset = random_split(train_dataset, [train_len, test_len])
#print(train_dataset.class_to_idx)
#print(train_dataset.imgs)

train_labels = torch.tensor(train_dataset.targets, dtype=torch.int64)
test_labels = torch.tensor(test_dataset.targets, dtype=torch.int64)



# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_labels, n_classes=n_classes, n_samples=n_samples)
test_batch_sampler = BalancedBatchSampler(test_labels, n_classes=n_classes, n_samples=n_samples)


if cuda_id is not None:
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
else:
    kwargs = {}

online_train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=my_collate, **kwargs)
online_test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=my_collate, **kwargs)


margin = 1.

#embedding_net = EmbeddingNet()
if frozen_net:
    embedding_net = FrozenPHOCnet(options.base_model)
    out_features = 256
else:
    embedding_net = torch.load(options.base_model)
    out_features = 648


model = embedding_net
if cuda_id is not None:
    if len(cuda_id) > 1:
        model = nn.DataParallel(model, device_ids=cuda_id)
        model.cuda()
    else:
        model.cuda(cuda_id[0])


#HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
## QUI DOVREMMO AGGIUNGERE UN NUOVO SELETTORE DI TRIPLETE PER COSTRUIRE LE NOSTRE TRIPLETTE
loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, sch_step, gamma=sch_gamma, last_epoch=-1)

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda_id, log_interval,out_features=out_features, save_model=SAVE_MODEL_NAME, metrics=[AverageNonzeroTripletsMetric()])
