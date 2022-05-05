import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import os
import random
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import myTransforms
from dataset.myDataset import my_collate
from dataset.myImageFolder import ImageFolderWithPaths

from wordSpotter import Spotter
from models.networks import FrozenPHOCnet

from tqdm import tqdm
import numpy as np
from operator import itemgetter


N_OF_SHOTS = 1

BASE_MODEL_NAME = 'weights/PHOC_best.pth'
TRAINED_MODEL_NAME = 'weights/PHOC_best_trained.pth'
MODEL_NAME = TRAINED_MODEL_NAME

ROW_FOLDER = "data/rows"
ALPHABET_FOLDER = "data/one_alph"

BACKGROUND_GRAY = 220
COLOR_SCORE_LINE = (0, 0, 255)
COLOR_SCORE_LINE_BASE = (150, 150, 150)

CUDA_IDS = None#[0]
cuda = torch.cuda.is_available()
if cuda:
    cuda_id = CUDA_IDS
else:
    cuda_id = None

def draw_score(dst_path_folder, row_path, ngram_path, score_dict):
    if not os.path.exists(dst_path_folder):
        os.makedirs(dst_path_folder)
    
    for ngram in score_dict:
        origin_img_name = row_path.split("/")[-1]
        ngram_img_name = ngram_path.split("/")[-1]
        new_img_name = origin_img_name.split('.')[0] +"_"+ ngram +"_" + ngram_img_name.split('.')[0] + "_." + origin_img_name.split('.')[-1]
        dst_path = os.path.join(dst_path_folder, new_img_name)

        row_img = cv2.imread(row_path)
        ngram_img = cv2.imread(ngram_path)

        ngram_img_full_height = ngram_img.shape[0] + 25
        ngram_img_full = np.ones((ngram_img_full_height,row_img.shape[1],3)) * BACKGROUND_GRAY
        ngram_img_full  = ngram_img_full.astype(np.uint8)

        ngram_img_full[10:ngram_img.shape[0]+10,10:ngram_img.shape[1]+10,:] = ngram_img
        cv2.putText(ngram_img_full, ngram, (30+ngram_img.shape[1], 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 5) # laber ngram
        cv2.putText(ngram_img_full, ngram_path, (30+ngram_img.shape[1], 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 5) # laber ngram

        score_img_height = row_img.shape[0] + 50
        score_img = np.ones((score_img_height,row_img.shape[1],3)) * BACKGROUND_GRAY
        score_img  = score_img.astype(np.uint8)

        list_score = score_dict[ngram]
        max_score = max(list_score, key=itemgetter(0))[0]
        min_score = min(list_score, key=itemgetter(0))[0]

        start = (0,score_img_height)
        report_max = min_score*1.15
        for score, x0, x1 in list_score:
            hg_line = int((score*score_img_height)/max_score)
            cv2.line(score_img, start, (x0, hg_line), COLOR_SCORE_LINE_BASE, 1)
            start = (x0, hg_line)
            if score < report_max:
                if hg_line <= 5:
                    hg_score = hg_line+10
                else:    
                    hg_score = hg_line-5 # +10
                cv2.line(score_img, (x0, hg_line), (x1, hg_line), COLOR_SCORE_LINE, 2)
                cv2.putText(score_img, f"{score:.2f}", (x0,hg_score), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_SCORE_LINE, 1)

        vis_concatenate = np.concatenate((ngram_img_full, row_img), axis=0)
        vis_concatenate = np.concatenate((vis_concatenate, score_img), axis=0)

        cv2.imwrite(dst_path, vis_concatenate)


def spot():
    if cuda_id is None:
        model = torch.load(MODEL_NAME, map_location='cpu')
    else:
        model = torch.load(MODEL_NAME)
        #freeze_phoc_model = FrozenPHOCnet(MODEL_NAME)
    #model = freeze_phoc_model

    spotter = Spotter(model, cuda_id=cuda_id)
    
    row_dataset = ImageFolderWithPaths(root=ROW_FOLDER,
                                    transform=transforms.Compose([
                                                myTransforms.toRGB(),
                                                myTransforms.Resize(),
                                                myTransforms.ToTensor(),
                                            ]))
    row_dataloader = DataLoader(row_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    row_dataiter = iter(row_dataloader)

    keyword_dataset = ImageFolderWithPaths(root=ALPHABET_FOLDER,
                                    transform=transforms.Compose([
                                                myTransforms.toRGB(),
                                                myTransforms.Resize(),
                                                myTransforms.ToTensor(),
                                            ]))
    kw_dataloader = DataLoader(keyword_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    ky_dataiter = iter(kw_dataloader)

    scores_all = {} #contiene tutte le liste di score calcolati

    for row_img, _, row_path in tqdm(row_dataiter):
        row_img = row_img[0]
        row_path = row_path[0]
        
        for cl_index in range(len(keyword_dataset.classes)):
            for shot in range(N_OF_SHOTS):
                kw_index = random.randint(0,len(keyword_dataset)-1)
                kw_img, cl, kw_path = keyword_dataset[kw_index]
                while cl != cl_index:
                    kw_index = random.randint(0,len(keyword_dataset)-1)
                    kw_img, cl, kw_path = keyword_dataset[kw_index]
                kw_transcript = kw_path.split(".")[0].split("/")[-2]
                
                scores = spotter(kw_transcript, kw_img, row_img)
                scores_all[(row_path,kw_path)] = scores
    
    return scores_all


if __name__ == "__main__":
    scores = spot()

    for ele in scores:
        draw_score("out/score_path", ele[0], ele[1], scores[ele])