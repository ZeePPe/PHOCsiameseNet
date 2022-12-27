import shutil
from statistics import mean
from string import capwords
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import os
import math
import random
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import myTransforms
from dataset.myDataset import my_collate
from dataset.myImageFolder import ImageFolderWithPaths

from ngram_retriver import Spotter
from models.networks import FrozenPHOCnet

from Levenshtein import distance as lev
from tqdm import tqdm
import numpy as np
from operator import itemgetter
from collections import Counter

import time
import re


N_OF_SHOTS = 10
THR_SCORE = 1.1
RESCORING_FACTOR = 5
DELETE_BOX_WHITE_THR = 50000
DELETE_BOX_SCORE_THR = 5
MIN_SCORE_QbS = 50


# density zone score computation flags
DENSITYZNE_SCORE_1 = True
DENSITYZNE_SCORE_2 = True
DENSITYZNE_SCORE_3a = False
DENSITYZNE_SCORE_3b = False
RESCORE_ORDER_NGRAM_STEP_3a = 2.5
MAX_DEPTH_RESCORE_3b = 3
RESCORE_BASE_3b = 1

CODING_STRING = "abcdefghijklmnopqrstuvwxyz"

OUT_FOLDER = "out/score_path"
BASE_MODEL_NAME = 'weights/PHOC_best.pth'
TRAINED_MODEL_NAME = 'weights/trained_001.pth' # 5pages Maria
TRAINED_MODEL_NAME ='weights/p10_01/trained_10p_bestvalloss_46.pth' # 10 pages frozen
TRAINED_MODEL_NAME ='weights/p10_01_nof/trained_10p_bestvalloss_7.pth' # 10 pages Not Frozen
MODEL_NAME = TRAINED_MODEL_NAME

ROW_FOLDER = "data/rows"
GT_FOLDER = "data/GT"
ALPHABET_FOLDER = "data/alphabet"
#ALPHABET_FOLDER = "data/limited_alphabet"
#ALPHABET_FOLDER = "data/one_alph" #fast

BACKGROUND_GRAY = 220
COLOR_SCORE_LINE = (0, 0, 255)
COLOR_SCORE_LINE_BASE = (150, 150, 150)

OUT_FILE_NAME = "words_position.txt"
OUT_FILE_NAME_ONLLYSCORE = "words_scores.txt"

CUDA_IDS = [0] #None#[0]
cuda = torch.cuda.is_available()
if cuda:
    cuda_id = CUDA_IDS
else:
    cuda_id = None

def draw_score(dst_path_folder, row_path, ngram_path, score_dict, th_score=THR_SCORE):
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
        report_max = min_score*th_score
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

def draw_score_2(dst_path_folder, row_path, ngram_label, list_score):
    if not os.path.exists(dst_path_folder):
        os.makedirs(dst_path_folder)
    
    origin_img_name = row_path.split("/")[-1]
    new_img_name = origin_img_name.split('.')[0] +"_" + ngram_label + "_." + origin_img_name.split('.')[-1]
    dst_path = os.path.join(dst_path_folder, new_img_name)

    row_img = cv2.imread(row_path)

    score_img_height = row_img.shape[0] + 50
    score_img = np.ones((score_img_height,row_img.shape[1],3)) * BACKGROUND_GRAY
    score_img  = score_img.astype(np.uint8)

    for score, x0, x1 in list_score:
        cv2.rectangle(score_img, (x0, 0), (x1, score_img_height), COLOR_SCORE_LINE, -1)
        cv2.putText(score_img, f"{ngram_label} {score:.2f}", (x0+10,score_img_height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    vis_concatenate = np.concatenate((row_img, score_img), axis=0)

    cv2.imwrite(dst_path, vis_concatenate)

def draw_density_zone(dst_path_folder, score_dict):
    if not os.path.exists(dst_path_folder):
        os.makedirs(dst_path_folder)

    curr_row_pat = None

    for (row_path, ngram_label), score in score_dict.items():
        if curr_row_pat != row_path:
            if curr_row_pat is not None:
                if not os.path.exists(os.path.dirname(dst_path)):
                    os.mkdir(os.path.dirname(dst_path))
                vis_concatenate = np.concatenate((row_img, score_img), axis=0)
                cv2.imwrite(dst_path, vis_concatenate)
            curr_row_pat = row_path
            origin_img_name = row_path.split("/")[-1]
            new_folder = row_path.split("/")[-2]
            new_img_name = origin_img_name.split('.')[0] +"_densZones." + origin_img_name.split('.')[-1]
            new_img_name = os.path.join(new_folder, new_img_name)
            dst_path = os.path.join(dst_path_folder, new_img_name)
            row_img = cv2.imread(row_path)

            score_img_height = row_img.shape[0] + 50
            score_img = np.ones((score_img_height,row_img.shape[1],3)) * BACKGROUND_GRAY
            score_img  = score_img.astype(np.uint8)

        for score, x0, x1 in score:
            cv2.rectangle(score_img, (x0, 0), (x1, score_img_height), COLOR_SCORE_LINE, -1)
            cv2.putText(score_img, f"{ngram_label} {score:.2f}", (x0+10,score_img_height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    vis_concatenate = np.concatenate((row_img, score_img), axis=0)

    cv2.imwrite(dst_path, vis_concatenate)

def draw_word_spotted(dst_path_folder, word_dict, min_score=MIN_SCORE_QbS, score_first=True):
    if not os.path.exists(dst_path_folder):
        os.makedirs(dst_path_folder)

    row_img = None

    for row_path, words in word_dict.items():
        origin_img_name = row_path.split("/")[-1]
        doc_folder = row_path.split("/")[-2]
        row_img = cv2.imread(row_path)
        score_img_height = row_img.shape[0] + 50

        for score, x0, x1, word in words:
            if score >= min_score:
                if score_first:
                    new_img_name = f"{score:.2f}_{origin_img_name.split('.')[0]}_WordsSpotted_{word}.{origin_img_name.split('.')[-1]}"
                else:
                    new_img_name = f"{origin_img_name.split('.')[0]}_WordsSpotted_{word}_{score:.2f}.{origin_img_name.split('.')[-1]}"
                if not os.path.exists(os.path.join(dst_path_folder, doc_folder)):
                    os.mkdir(os.path.join(dst_path_folder, doc_folder))
                dst_path = os.path.join(dst_path_folder, doc_folder, new_img_name)

                score_img = np.ones((score_img_height,row_img.shape[1],3)) * BACKGROUND_GRAY
                score_img  = score_img.astype(np.uint8)

                cv2.rectangle(score_img, (x0, 0), (x1, score_img_height), COLOR_SCORE_LINE, -1)
                cv2.putText(score_img, f"{word} {score:.2f}", (x0+10,score_img_height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                vis_concatenate = np.concatenate((row_img, score_img), axis=0)

                cv2.imwrite(dst_path, vis_concatenate)

def save_word_spotted(dst_path_folder, word_dict, min_score=MIN_SCORE_QbS):
    """
    Save the file of the spotted words
    """
    if not os.path.exists(dst_path_folder):
        os.makedirs(dst_path_folder)

    for row_path, words in word_dict.items():
        origin_img_name = row_path.split("/")[-1]
        doc_folder = row_path.split("/")[-2]

        curr_out_folder = os.path.join(dst_path_folder, doc_folder)
        if not os.path.exists(curr_out_folder):
            os.mkdir(curr_out_folder)

        for score, x0, x1, word in words:
            with(open(os.path.join(curr_out_folder, OUT_FILE_NAME), "a") as out_file):
                if os.path.getsize(os.path.join(curr_out_folder, OUT_FILE_NAME)) == 0:
                    out_file.write("N_ROW,WORD,X0,X1,SCORE\n")

                if score >= min_score:
                    out_file.write(f"{origin_img_name.split('.')[0]},{word},{x0},{x1},{score}\n")
            
            with(open(os.path.join(curr_out_folder, OUT_FILE_NAME_ONLLYSCORE), "a") as out_file):
                if score >= min_score:
                    out_file.write(f"{score}\n")
        
        GTfile_path = get_gt_filename(row_path)
        with(open(GTfile_path, "r") as GT_file):
            gt = GT_file.readline().rstrip()
        gt = re.sub(r'[^\w\s]', ' ', gt)
        gt = re.sub(' +', ' ', gt)
        gt = gt.rstrip()
        words_in_gt = gt.split(" ")
        n_words = words_in_gt.count(word)

        with(open(os.path.join(curr_out_folder, OUT_FILE_NAME), "a") as out_file):
            out_file.write(f"GT  --> line:'{origin_img_name.split('.')[0]}' n_word={n_words}\n\n")
    
    gt_folder = os.path.join(GT_FOLDER, doc_folder)
    tot_in_page = 0
    for file in os.listdir(gt_folder):
        with(open(os.path.join(gt_folder, file), "r") as GT_file):
            gt = GT_file.readline().rstrip()
            words_in_gt = gt.split(" ")
            n_words = words_in_gt.count(word)
            if n_words > 0:
                tot_in_page += n_words

    with(open(os.path.join(curr_out_folder, OUT_FILE_NAME), "a") as out_file):
        out_file.write(f"\n\n\nGT TOTAL  --> word:'{word}' items={tot_in_page}")

def get_gt_filename(rowpath):
    row = int(rowpath.split("/")[-1].split(".")[0])
    row -= 1
    row = str(row).rjust(2, "0")
    doc = rowpath.split("/")[-2]
    filename = f"gt_{row}_{doc}.txt"
    gt_path = os.path.join(GT_FOLDER, doc, filename)
    return gt_path
                    

                

# §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

def spot(ngrams_list=None, row_folder=ROW_FOLDER, alphabet_folder=ALPHABET_FOLDER, n_shots=N_OF_SHOTS, cuda_id=cuda_id):
    """
    spot the N-grams present in list ngram:list in the lextlines
    Return the list of scores, where a score is a list of all distances between the support ngram image
    and the text line image.
    If ngrams_list is None, it test all classes in the alphabet_folder
    """
    if cuda_id is None:
        model = torch.load(MODEL_NAME, map_location='cpu')
    else:
        model = torch.load(MODEL_NAME)
        #freeze_phoc_model = FrozenPHOCnet(MODEL_NAME)
    #model = freeze_phoc_model

    spotter = Spotter(model, cuda_id=cuda_id)
    
    row_dataset = ImageFolderWithPaths(root=row_folder,
                                    transform=transforms.Compose([
                                                myTransforms.toRGB(),
                                                myTransforms.Resize(),
                                                myTransforms.ToTensor(),
                                            ]))
    row_dataloader = DataLoader(row_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    row_dataiter = iter(row_dataloader)

    keyword_dataset = ImageFolderWithPaths(root=alphabet_folder,
                                    transform=transforms.Compose([
                                                myTransforms.toRGB(),
                                                myTransforms.Resize(),
                                                myTransforms.ToTensor(),
                                            ]))
    #kw_dataloader = DataLoader(keyword_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    #ky_dataiter = iter(kw_dataloader)

    if ngrams_list is None:
        ngrams_list = keyword_dataset.classes
    
    class_list = []
    for index, ngram in enumerate(keyword_dataset.classes):
        if ngram in ngrams_list:
            class_list.append(index)

    scores_all = {} #contiene tutte le liste di score calcolati

    for row_img, _, row_path in tqdm(row_dataiter):
        row_img = row_img[0]
        row_path = row_path[0]
        
        #for cl_index in range(len(keyword_dataset.classes)):
        for cl_index in class_list:
            idices_for_curr_class = [idx for idx, value in enumerate(keyword_dataset.targets) if value == cl_index]

            for shot in range(n_shots):
                kw_index = random.choice(idices_for_curr_class)
                kw_img, cl, kw_path = keyword_dataset[kw_index]
                kw_transcript = kw_path.split(".")[0].split("/")[-2]
                scores = spotter(kw_transcript, kw_img, row_img)
                scores_all[(row_path,kw_path)] = scores
    
    return scores_all

def filter(scores, th_score=THR_SCORE):
    """
    The function filters the spotting output.
    all the output for a same class are fused. When there is an overlapping of maxima,
    the score for the overlapping is recalculated decreasing the result score
    """
    maxima = {} # key = (row_path, ngram_label)

    set_ngram_images = set()

    # get only the maxima for each support n-gram 
    for (row_path, ngram_path), dic_scores in scores.items():
        if (row_path, ngram_path) not in set_ngram_images:
            set_ngram_images.add((row_path, ngram_path))
            for ngram_key, current_scores in dic_scores.items():
                
                min_score = min(current_scores, key=itemgetter(0))[0]
                report_max = min_score*th_score

                all_max_scores = [box for box in current_scores if box[0] <= report_max]

                try:
                    maxima[(row_path, ngram_key)] = maxima[(row_path, ngram_key)] + (all_max_scores)
                except:
                    maxima[(row_path, ngram_key)] = all_max_scores

    #overlapp filter
    for (row_path, ngram_key), score in maxima.items():
        sorted_by_position = sorted(score, key=lambda box: box[1])
        overlapping_set = []

        box_top = None
        for index_current in range(len(sorted_by_position)):    
            box = sorted_by_position[index_current]
            
            if box_top is None:
                box_top = box
            else:
                if overlapping_box(box_top, box):
                    new_score_l = min(box_top[0], box[0])-0.25*(abs(box_top[0]-box[0]))
                    min_sc = min(box_top[0], box[0])
                    sigm = abs(box_top[0]-box[0])/(1+abs(box_top[0]-box[0]))
                    new_score = min_sc - RESCORING_FACTOR*((3/4)**min_sc*(1-sigm))
                    if new_score < 0:
                        new_score = 0
                    box_top = (new_score,
                               int((box_top[1]+box[1])/2),
                               int((box_top[2]+box[2])/2))
                else:
                    overlapping_set.append(box_top)
                    box_top = None

            if index_current == len(sorted_by_position)-1 and box_top is not None:
                overlapping_set.append(box_top)

        min_score = min(overlapping_set, key=itemgetter(0))[0]
        if min_score < 0:
            for index in range(len(overlapping_set)):
                box = overlapping_set[index]
                overlapping_set[index] = (box[0] - min_score, box[1], box[2])
        maxima[(row_path, ngram_key)] = overlapping_set
    

    # delete boxes where not "enough" ink and with score to hight
    for (row_path, ngram_label), boxes in maxima.items():
        row_image = cv2.imread(row_path, cv2.IMREAD_GRAYSCALE)
        row_image = cv2.bitwise_not(row_image)

        for box in boxes:
            ink = np.sum(row_image[0:row_image.shape[0], box[1]:box[2]])
            if ink < DELETE_BOX_WHITE_THR:
                boxes.remove(box)
                break
            
            if box[0] > DELETE_BOX_SCORE_THR:
                boxes.remove(box)

    return maxima

def get_density_zones(scores):
    density_zones = {}

    # step (1) - densit zone in a vector of zeros for each row;
    #            in the vector there are 1s only in the densty zone 
    for (row_path, ngram_label), score in scores.items():
        if row_path not in density_zones:
            row_img = cv2.imread(row_path)
            density_zones[row_path] = np.zeros(row_img.shape[1], dtype=np.uint8)
        
        for box in score:
            density_zones[row_path][box[1]:box[2]]=1
    
    # step (2) - differentiate de zero vectors and keep only the boundary
    density_dict_rows = {}
    density_boundary_rows = {}
    for row_path, all_zones in density_zones.items():
        diff_dz = np.diff(all_zones)
    
        density_boundaries = np.where(diff_dz != 0)[0]
        if density_boundaries.shape[0] % 2 != 0:
            print("sono dispari??")

        density_boundaries = np.reshape(density_boundaries, (-1, 2))

        density_dic = {}
        for zone in density_boundaries:
            density_dic[(zone[0], zone[1])] = []
        density_dict_rows[row_path] = density_dic
        density_boundary_rows[row_path] = density_boundaries

    for (row_path, ngram_label), score in scores.items():
        density_boundaries = density_boundary_rows[row_path]
        density_dic = density_dict_rows[row_path]
        for box in score:
            for zone in density_boundaries:
                if zone[0] <= box[1] <= zone[1]:
                    density_dic[(zone[0], zone[1])].append((box,ngram_label))
        density_zones[row_path] = density_dic

    return density_zones

def score_density_zones(density_zones, queryWord, min_score=MIN_SCORE_QbS):
    """
    Scores the density zones to provide the output of the KWS QbS.

    the score is computed: 
       - (1) high number of ngram present
       - (2) avarage value score ngrams present
       - (3) order of ngrams in a zone 
             exist a sequence of n-grams 'close' to the true sequenze (gt sequence) 

    out is a dictionary with key the row_path and value a list of (score, x0, x1, queryWord)
    score is a number between 0 and 100, higher the value better is the probability for the zone to be a spotted zone
    """
    out_dic = {}

    bigrams = get_ngrams_list(queryWord, n_min=2, n_max=2)
    trigrams = get_ngrams_list(queryWord, n_min=3, n_max=3)

    ngrams_number = len(bigrams) + len(trigrams)

    for row_path, den_zones in density_zones.items():
        if len(den_zones) > 0:
            out_dic[row_path] = []

            for zone, boxes in den_zones.items():
                x0, x1 = zone[0], zone[1]
                score_zone = 100
                # score (1):
                if DENSITYZNE_SCORE_1:
                    score_zone = score_zone * (len(boxes)/ngrams_number) 
                    #if score_zone > 100: score_zone = 100
                
                # score (2):
                if DENSITYZNE_SCORE_2:
                    avg_score =  mean(box[0][0] for box in boxes)
                    score_zone = score_zone - avg_score
                
                # score (3)
                if DENSITYZNE_SCORE_3a:
                    if score_zone > min_score:
                        ordered_boxes = sorted(boxes, key=lambda box: box[0][1])

                        optimal_ind = ""
                        pos_str = ""
                        position_ind = []

                        for i in range(len(queryWord)-1):
                            cur_bi = queryWord[i:i+2]
                            cur_tri = queryWord[i:i+3]
                            if cur_bi in os.listdir(ALPHABET_FOLDER) or cur_tri in os.listdir(ALPHABET_FOLDER):
                                optimal_ind += CODING_STRING[i]
                    
                        for _, ngram in ordered_boxes:
                            pos_str += CODING_STRING[queryWord.index(ngram)]

                        # clean multiple adiacent
                        box_to_fuse=[]
                        for i in range(len(pos_str)-1):
                            if pos_str[i] == pos_str[i+1]:
                                box_to_fuse.append(i)
                        for i_tofuse in reversed(box_to_fuse):
                            pos_str = pos_str[:i_tofuse]+pos_str[i_tofuse+1:]
                            if i_tofuse < len(ordered_boxes)/2:
                                del ordered_boxes[i_tofuse+1]
                            else:
                                del ordered_boxes[i_tofuse]

                        for ind_x0 in range(len(pos_str)-1):
                            for ind_x1 in range(ind_x0+1, len(pos_str)+1):
                                position_ind.append((ind_x0, ind_x1-1, lev(optimal_ind, pos_str[ind_x0:ind_x1])))
        
                        if len(pos_str) == 1:
                            best_alignment = (0,0,0)
                        else:
                            best_alignment = min(position_ind, key=lambda el: (el[-1], -el[1]-el[0]))

                        #x0 = ordered_boxes[best_alignment[0]][0][1]
                        #x1 = ordered_boxes[best_alignment[1]][0][2]

                        score_zone -= best_alignment[-1]*RESCORE_ORDER_NGRAM_STEP_3a

                if DENSITYZNE_SCORE_3b:
                    if score_zone > min_score:
                        ordered_boxes = sorted(boxes, key=lambda box: box[0][1])

                        optimal_ind = ""
                        pos_str = ""
                        position_ind = []

                        for i in range(len(queryWord)-1):
                            cur_bi = queryWord[i:i+2]
                            cur_tri = queryWord[i:i+3]
                            if cur_bi in os.listdir(ALPHABET_FOLDER) or cur_tri in os.listdir(ALPHABET_FOLDER):
                                optimal_ind += CODING_STRING[i]
                    
                        for _, ngram in ordered_boxes:
                            pos_str += CODING_STRING[queryWord.index(ngram)]

                        if len(queryWord) <= 4:
                            cardinality = 2
                        elif len(queryWord) <= 6:
                            cardinality = 3
                        else:
                            cardinality = MAX_DEPTH_RESCORE_3b
                        
                        common_factors = []
                        for curr_card in range(2,cardinality+1):
                            seqs_zone = _divide_substrings(pos_str, curr_card)
                            seqs_query = _divide_substrings(optimal_ind, curr_card)
                            
                            for seq_zone, seq_query in zip(seqs_zone, seqs_query):
                                common_letters = Counter(seq_zone) & Counter(seq_query)
                                common_letters = sum(common_letters.values())
                                
                                if seq_query != "":
                                    common_factors.append(common_letters/len(seq_query))
                        
                        mean_refactor = mean(common_factors)

                        score_zone = score_zone*(RESCORE_BASE_3b*mean_refactor)
                        

                # if score_zone > 100:
                #     score_zone = 100
                out_dic[row_path].append((score_zone, x0, x1, queryWord))
    
    return out_dic
 
def _divide_substrings(string, cardinality):
    out_list=[]
    step = math.ceil(len(string) * 1/cardinality)
    if step == 0:
        step = 1
    end = step
    start = 0
    for _ in range(cardinality):
        out_list.append(string[start:end])
    
        start = end
        end += step
        #if end+step > len(string):
        #    end += step
    
    return out_list

def overlapping_box(box1, box2) -> bool:
    if (box2[1] <= box1[1] <= box2[2]) or (box1[1] <= box2[1] <= box1[2]):
        return True
    return False

def get_ngrams_list(keyword, n_min=2, n_max=3):
    """
    returns the list of N-grams of the keyword included in the selected alphabet
    nmax define the maximum number of characters for N-gram ()
    """
    ngrams_list= []
    for n in range(n_min,n_max+1):
        for index in range(len(keyword)-n+1):
            ngram = keyword[index:index+n]

            if ngram in os.listdir(ALPHABET_FOLDER):
                ngrams_list.append(ngram)

    return ngrams_list


if __name__ == "__main__":
    # compute the spotting
    # 350_320_001
    #all_query = ["examination", "finitive","com","defini","motion","perform","pronounced","puts","requisition", "sources"] # OOV
    #all_query = ["being", "either", "formed", "more", "such", "than", "that", "this", "with"]
    #all_query = ["examination", "finitive","com","defini","motion","perform","pronounced","puts","requisition", "sources", "being", "either", "formed", "more", "such", "than", "that", "this", "with"]
    # 350_321_001
    #all_query = ["absolute","applied","exhibited","incidental","original","shall","special","those","Trial","whether"] # OOV
    #all_query = ["been","evidence","more","signed","this","time","which"]
    #all_query = ["absolute","applied","exhibited","incidental","original","shall","special","those","Trial","whether", "been","evidence","more","signed","this","time","which"]
    ## 350_322_001
    #all_query = ["Appellate","appoint","attendance","both","called","effect","hearing","performance","station","Visitors"] # OOV
    #all_query = ["during","from","have","place","question","same","take","that","this","with"] # IV
    #all_query = ["during","from","have","place","question","same","take","that","this","with","Appellate","appoint","attendance","both","called","effect","hearing","performance","station","Visitors"]
    #all_query = ["called","effect","hearing","performance","station","Visitors"]
    ## 035_323_001
    #all_query = ["active","ance","aptitude","give","giving","great","greatest","looked","ordinary","titude"] # OOV
    #all_query = ["different","least","number","purpose","that","this","twice"]
    #all_query = ["active","ance","aptitude","give","giving","great","greatest","looked","ordinary","titude","different","least","number","purpose","that","this","twice"] # OOV

    #035_324_001
    #all_query = ["apinative","auditive","body","capable","exercised","Exercised","pronounced","several"] #OOV
    #all_query = ["each","have","individual","only","same","such","than","they","with"]
    #all_query = ["apinative","auditive","body","capable","exercised","Exercised","pronounced","several","each","have","individual","only","same","such","than","they","with"]

    #035_325_001
    #all_query = ["actions","actors","signified","experience","interrogative","remarks","inspection","occasion","otherwise","theatre"] # oov
    #all_query = ["this","other","written","evidence","expressed","exercise","themselves","persons","portions","take"]
    #all_query = ["actions","actors","signified","experience","interrogative","remarks","inspection","occasion","otherwise","theatre","this","other","written","evidence","expressed","exercise","themselves","persons","portions","take"]

    #035_327_001
    #all_query = ["ation","deemed","regarded","misdecision","defendant","indication","concurrence","legislature","composed","immediate"]#OOV
    #all_query = ["which","these","place","appellation","there"]
    #all_query = ["ation","deemed","regarded","misdecision","defendant","indication","concurrence","legislature","composed","immediate","which","these","place","appellation","there"]

    ##035_328_001
    #all_query = ["cision","them","comparative","procedure","affenders","multiplicity","otherwise","function","registration","appellate"] #OOv
    #all_query = ["other","office","exercise","given","should","part","these"]
    #all_query = ["other","office","exercise","given","should","part","these","cision","them","comparative","procedure","affenders","multiplicity","otherwise","function","registration","appellate"]

    ##071_002_002
    #all_query = ["before","beats","assaulted","keeping","concerning","accidents","explosion","compensation","navigable","receive"]# OOV
    #all_query = ["against","which","cases","persons","from","make","made"]
    #all_query = ["against","which","cases","persons","from","make","made","before","beats","assaulted","keeping","concerning","accidents","explosion","compensation","navigable","receive"]

    ##071_002_003
    #all_query = ["states","since","them","enter","various","neither","concerning","according","several","provinces"] # oov
    #all_query = ["that","persons","respect","upon","above","question","universal","different","mentioned","other"]
    #all_query = ["that","persons","respect","upon","above","question","universal","different","mentioned","other","states","since","them","enter","various","neither","concerning","according","several","provinces"]

    ##071_003_003
    #all_query = ["officer","guardian","domestic","family","article","another","children","separate","preceptor","adults"]#oov
    #all_query = ["which","note","same","account","persons","have","these","time"]
    #all_query = ["which","note","same","account","persons","have","these","time","officer","guardian","domestic","family","article","another","children","separate","preceptor","adults"]
    
    ##071_003_004
    #all_query = ["thing","perfomed","pursuance","military","service","without","occasion","performed","execution","process"] #oov
    #all_query = ["person","either","that","regard","other","upon","which","during","time","such"]
    #all_query = ["person","either","that","regard","other","upon","which","during","time","such","thing","perfomed","pursuance","military","service","without","occasion","performed","execution","process"]

    ##071_008_004
    #all_query = ["intention","them","round","within","particular","frequented","instance","another"]#oov
    #all_query = ["that","person","upon","have","question","beginning","time","this"]
    #all_query = ["that","person","upon","have","question","beginning","time","this","intention","them","round","within","particular","frequented","instance","another"]

    ##071_010_002
    #all_query = ["exist","circum","where","matter","concerning","established","supposition","probable","supposing"] # oov
    #all_query = ["which","existence","question","material","respect"]
    #all_query = ["which","existence","question","material","respect","exist","circum","where","matter","concerning","established","supposition","probable","supposing"]

    ##A116_405_004
    all_query = ["addition","daily","nusance","without","liverystables","registered","passengers","extinguished","relighting","avenues"] #oov
    #all_query = ["should","which","person","different","account","purpose","taken"]
    #all_query = ["should","which","person","different","account","purpose","taken","addition","daily","nusance","without","liverystables","registered","passengers","extinguished","relighting","avenues"]
    #all_query = ["purpose","taken","addition","daily","nusance","without","liverystables","registered","passengers","extinguished","relighting","avenues"]
    
    # out folder
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

    for query in all_query:
        print(f"spotting '{query}'")

        ngram_list = get_ngrams_list(query)

        start_time = time.time()
        
        scores = spot(ngram_list)

        ex_time = time.time() - start_time
        print(f"'{query}' K={N_OF_SHOTS} spotted in {ex_time} Sec")
        
        # filter the spotting results obtaining only the maxima
        filtered_scores = filter(scores)
        ex_time = time.time() - start_time
        print(f"'{query}' filtered in {ex_time} Sec")

        #find density zones
        density_zones = get_density_zones(filtered_scores)
        ex_time = time.time() - start_time
        print(f"'{query}' density zones in {ex_time} Sec")

        #Score density zones -> KWS
        kws_result = score_density_zones(density_zones, query)
        ex_time = time.time() - start_time
        print(f"'{query}' KWS results in {ex_time} Sec")

        # OUTPUT IMAGES --------------------------------------------------------->
        out_dir = os.path.join(OUT_FOLDER, query)
        os.mkdir(out_dir)

        # for ele in scores:
        #    draw_score(out_dir, ele[0], ele[1], scores[ele])
        
        #for ele, scr in filtered_scores.items():
        #    draw_score_2(out_dir, ele[0], ele[1], scr)
        
        #draw_density_zone(out_dir, filtered_scores)

        draw_word_spotted(out_dir, kws_result, min_score=MIN_SCORE_QbS)

        save_word_spotted(out_dir, kws_result, min_score=MIN_SCORE_QbS)


    print("Done!")