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
import pickle

from models.trainer import fit

from dataset import myTransforms
from dataset.myImageFolder import ImageFolderWithPaths, HardPositiveBatchSampler
from dataset.myDataset import my_collate, MyImagesDS

#from models.datasets import BalancedBatchSampler

# Set up the network and training parameters
from models.networks import EmbeddingNet
from models.losses import OnlineTripletLoss
from models.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from models.HardPositiveHardNegativeTripletSelector import HardPositiveHardNegativeTripletSelector
from models.metrics import AverageNonzeroTripletsMetric
from models.networks import FrozenPHOCnet
from train_config import getTrainOptions

from hard_triplets_utils import separate_train_dataset, HardTripletSelector, edit_distance_hard_negative
import config as C

print(" --- Training Siamese N-Gram network ---")
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

print(f"\n\nTrain Info")
print(f"Training path:{options.training_path}\nTest path:{options.test_path}")
print(f"Num classes:{n_classes}\nNum samples:{n_samples}")
print(f"Learning Rate:{lr}\nMax Epoch:{n_epochs}\n log interval:{log_interval}")
print(f"Scheduler step:{sch_step} scheduler gamma:{sch_gamma}")
print(f"Frozen Model:{frozen_net}")
print(f"GPUs:{cuda}")
print(f"\nSave Model:{options.save_model}")

train_dataset = ImageFolderWithPaths(root=options.training_path,
                                transform=transforms.Compose([
                                            myTransforms.toRGB(),
                                            myTransforms.Resize(),
                                            myTransforms.ToTensor(),
                                        ]))
test_dataset = ImageFolderWithPaths(root=options.training_path,
                                transform=transforms.Compose([
                                            myTransforms.toRGB(),
                                            myTransforms.Resize(),
                                            myTransforms.ToTensor(),
                                        ]))

with open(C.EMBEDDING_FILE, "rb") as embeddings_file:
        embeddings = pickle.load(embeddings_file)

class_to_idx = train_dataset.class_to_idx
# maximum number of samples we want to use in the train subset
n_samples = 8
# split the dataset in train_subset and validation_subset
train_subset, validation_subset = separate_train_dataset(train_dataset, n_samples, embeddings)  
print(type(train_subset))
# Sono delle liste????
# dovrebbero essere dei tipi dataset!!!!

train_labels = torch.tensor(train_dataset.targets, dtype=torch.int64)
test_labels = torch.tensor(test_dataset.targets, dtype=torch.int64)

# Creiamo dei batches considerando tutte le classi disponibili e campionando soltanto i top elements per
# per ogni classe prendendo 3 elementi per classe (quando disponibili)
# MODIFICARE QUESTO PER METTERE GLI ELEMENTI CHE CI INTERESSANO - Mario
train_batch_sampler = HardPositiveBatchSampler(train_labels, n_classes=n_classes, n_samples=3)
test_batch_sampler = HardPositiveBatchSampler(test_labels, n_classes=n_classes, n_samples=3)

if cuda_id is not None:
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
else:
    kwargs = {}

online_train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=my_collate, **kwargs)
online_test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=my_collate, **kwargs)


# caricamento del modello per il calcolo degli embedding ------------------------------------------------------------
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

margin = 1.

# metriche -> selettore triplette
#   HardestNegativeTripletSelector(margin)  ->
#   RandomNegativeTripletSelector(margin)   ->
#   SemihardNegativeTripletSelector(margin) -> 
#   HardPositiveHardNegativeTripletSelector()  -> Mario selector
#    HardTripletSelector(train_subset, edit_distance_hard_negative, class_to_idx) -> Mario


triplet_selector = SemihardNegativeTripletSelector(margin)

#triplet_selector = HardTripletSelector(train_subset, edit_distance_hard_negative, class_to_idx)
#triplets = triplet_selector.get_triplets()
# il metodo detrno la loss prevede triplets = self.triplet_selector.get_triplets(embeddings, target)
# i batch sono insiemi di triplette!


loss_fn = OnlineTripletLoss(margin, triplet_selector)
# in alternativa possiamo usare TripletLoss(anchor, positive, negative) che lavora su una singola tripletta
# ma sarebbe meglio lavorare su batches
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, sch_step, gamma=sch_gamma, last_epoch=-1)

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda_id, log_interval, out_features=out_features, save_model=options.save_model, metrics=[AverageNonzeroTripletsMetric()])
