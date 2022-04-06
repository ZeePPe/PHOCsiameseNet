import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseMeasurer:
    def __init__(self, model, cuda_id=None):
        self.model = model
        self.cuda_id = cuda_id

        if cuda_id is not None:
            if len(cuda_id) > 1:
                self.model = nn.DataParallel(self.model, device_ids=cuda_id)
                self.model.cuda()
            else:
                self.model.cuda(cuda_id[0])
        
        self.model.eval()


    def get_distance(self, img1, img2):
        img1 = img1[None, :]
        img2 = img2[None, :]

        if self.cuda_id is not None:
            if len(self.cuda_id) > 1:
                img1 = img1.cuda()
                img2 = img2.cuda()
            else:
                img1 = img1.cuda(self.cuda_id[0])
                img2 = img2.cuda(self.cuda_id[0])
        
        phoc_rep1 = self.model(img1)
        phoc_rep2 = self.model(img2)
        if type(phoc_rep1) is dict:
            phoc_rep1 = phoc_rep1['phoc'][-1]
        if type(phoc_rep2) is dict:
            phoc_rep2 = phoc_rep2['phoc'][-1]

        euclidean_distance = F.pairwise_distance(phoc_rep1, phoc_rep2)
        #dist = (phoc_rep1 - phoc_rep2).pow(2).sum().sqrt()
        
        return euclidean_distance.item()

    def get_embedding(self, img):

        img = img[None, :]

        if self.cuda_id is not None:
            if len(self.cuda_id) > 1:
                img = img.cuda()
            else:
                img = img.cuda(self.cuda_id[0])

        phoc_rep = self.model(img)
        if type(phoc_rep) is dict:
            phoc_rep = torch.sigmoid(phoc_rep['phoc'] )

        return phoc_rep
