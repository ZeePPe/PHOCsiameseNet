from siamese_measurer import SiameseMeasurer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import os
import cv2

class Spotter(nn.Module):
    """
    model is the model uset to compute the distance
    threshold is the minimum distance to consider as good score, if not defined there is no threshold
    """
    def __init__(self, model, threshold=-1, cuda_id=None) -> None:
        super(Spotter, self).__init__()
        self.model = model
        self.threshold = threshold
        self.measurer = SiameseMeasurer(model, cuda_id)
    
    
    """ Search kw_image in the row_image using the self.model to compute the distance
    it returns: {"Ngram_transcript":[(score, x0,x1)]}
    it is a dictionary where key is the transcript of keyword image and the value is a list where:
        score - is the distance of the keyword
        x0,x1 - is the bounding box in the row_image, x0 is the starting index, x1 the end index 
    """
    def forward(self,kw_transcript, kw_image, row_img):
        
        # trova dove inizia la scritta, dove quindi non c'è bianco
        # taglia l'immagine della dim della kw_img
        # calcola la distanza
        # fai scorrere la finestra di mezzo carattere

        score_list = []
        self.measurer.set_base_representation(kw_image)

        window_width = kw_image.shape[-1]
        shift = round(window_width/(len(kw_transcript)*2))

        curr_x0 = 0
        while curr_x0 < row_img.shape[-1]:
            col = row_img[:,:,curr_x0]
            sum_pixel = torch.sum(col)
            
            if sum_pixel != 0:
                #print(curr_x0, sum_pixel.item())
                curr_x1 = curr_x0+window_width
                if curr_x1 > row_img.shape[-1]:
                    curr_x1 = row_img.shape[-1]
                row_crop = row_img[:,:,curr_x0:curr_x1]

                score = self.measurer.get_distance_fast(row_crop)

                score_list.append((score, curr_x0, curr_x1))

                curr_x0 += shift
            curr_x0 += 1
        
        return {kw_transcript: score_list}
