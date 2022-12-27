import os
import shutil
import re
import string

"""
calcola le statistiche sulle parole OOV presenti nel ts e ts

"""
OUT_FOLDER = "outstats/cose"
BASE_MODEL_NAME = 'weights/PHOC_best.pth'
TRAINED_MODEL_NAME = 'weights/trained_001.pth'
MODEL_NAME = TRAINED_MODEL_NAME

ROW_FOLDER = "data/rows"
GT_FOLDER = "data/GT"
GT_TRAINING_FOLDER = "data/GT_TRAINING"
ALPHABET_FOLDER = "data/alphabet_augmented"
#ALPHABET_FOLDER = "data/limited_alphabet"
#ALPHABET_FOLDER = "data/one_alph" #fast

BACKGROUND_GRAY = 220
COLOR_SCORE_LINE = (0, 0, 255)


def count_apphabet_items(min_items=3):
    n_class = 0
    n_items = 0
    min_5 = 0
    max_item_perclass = 0
    min_item_perclass = float('inf')

    for class_folder in os.listdir(ALPHABET_FOLDER):
        n_class += 1
        curr_item = 0
        for item in os.listdir(os.path.join(ALPHABET_FOLDER, class_folder)):
            curr_item += 1
        
        if curr_item > max_item_perclass:
            max_item_perclass = curr_item
        
        if curr_item < min_item_perclass:
            min_item_perclass = curr_item

        if curr_item < min_items:
            min_5 += 1
        
        n_items += curr_item

    return n_class, n_items, max_item_perclass, min_item_perclass, min_5


def count_words(folder):
    all_words = {}
    for doc_folder in os.listdir(folder):
        for line_file_name in os.listdir(os.path.join(folder, doc_folder)):
            with(open(os.path.join(folder, doc_folder, line_file_name), "r") as gt_file):
                line = gt_file.readline()
                #line = re.sub(r'[^\w\s]', '', line)
                line = line.translate(str.maketrans('', '', string.punctuation))
                line = re.sub(' +', ' ', line)
                line = line.strip()
                line = line.lower()

                for word in line.split(" "):
                    if word in all_words:
                        all_words[word] += 1
                    else:
                        all_words[word] = 1
    return all_words

def ngram_raprresentation(all_words, set_ngrams):
    searchable = []
    not_searchable = []
    for word in all_words:
        if ngram_word_raprresentations(word, set_ngrams):
            searchable.append(word)
        else:
            not_searchable.append(word)
                
    return searchable, not_searchable

def ngram_word_raprresentations(word, set_ngrams, overlapp=True):
    """
    return True if a word is searchable, i.e. the word is obtainabla as a composition of alphabet n-grams
    if overlapp = false, it is considered olso the first n-gram that not overlapp the current:
        Nature -> 'Na' 'tu'
    """
    ind = 1
    start_bi = word[0:2]
    start_tri = word[0:3]

    if start_bi in set_ngrams:
        while ind < len(word)-1:
            bi_t1 = word[ind:ind+2]
            bi_t2 = word[ind+1:ind+3]
            tri_t1 = word[ind:ind+3]
            tri_t2 = word[ind+1:ind+4]

            if len(bi_t1) == 2 and bi_t1 in set_ngrams:
                ind += 1
            elif not overlapp and len(bi_t2) == 2 and bi_t2 in set_ngrams:
                ind += 2
            elif len(tri_t1) == 3 and tri_t1 in set_ngrams:
                ind += 1
            elif not overlapp and len(tri_t2) == 3 and tri_t2 in set_ngrams:
                ind += 2
            else:
                return False

    elif start_tri in set_ngrams:
        while ind < len(word)-1:
            bi_t1 = word[ind:ind+2]
            bi_t2 = word[ind+1:ind+3]
            bi_t3 = word[ind+2:ind+4]
            tri_t1 = word[ind:ind+3]
            tri_t2 = word[ind+1:ind+4]
            tri_t3 = word[ind+2:ind+5]

            if len(bi_t1) == 2 and bi_t1 in set_ngrams:
                ind += 1
            elif len(bi_t2) == 2 and bi_t2 in set_ngrams:
                ind += 2
            elif not overlapp and len(bi_t3) == 3 and bi_t2 in set_ngrams:
                ind += 3
            elif len(tri_t1) == 3 and tri_t1 in set_ngrams:
                ind += 1
            elif len(tri_t2) == 3 and tri_t2 in set_ngrams:
                ind += 2
            elif not overlapp and len(tri_t3) == 3 and tri_t3 in set_ngrams:
                ind += 3
            else:
                return False
    else:
        return False

    return True

def save_on_file(all_word, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.mkdir(os.path.dirname(outfile))

    with(open(outfile, "w") as file):
        if isinstance(all_word, dict):
            all_word =  {k: v for k, v in sorted(all_word.items(), key=lambda item: item[1], reverse=True)}
        
            for word, val in all_word.items():
                file.write(f"{word}: {val}\n")
        else:
            for word in all_word:
                file.write(f"{word}\n")


if __name__ == "__main__":
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.mkdir(OUT_FOLDER)

    oov_searched = ["finivive","examination","com","defini","motion","perform","pronounced","puts","requisition","sources","absolute","applied","exhibited","incidental","original","shall","special","those","trial","whetehr","applied","exhibited","incidental","original","shall","those","trial","whetehr","active","aptitude","give","giving","great","greatest","looked","ordinary","titude","apinative","auditive","capable","exercised","Exercised","pronounced","several","actions","signified","experience","remarks","inspection","occasion","otherwise","theatre","action","deemed","defendant","indication","concurrence","composed","cision","them","comparative","affenders","multiplicity","otherwise","registration","before","assaulted","keeping","concerning","accidents","compensation","navigable","receive","states","since","them","various","neither","concerning","according","provinces","thing","perfomed","pursuance","military","service","without","occasion","execution","process","intention","them","round","within","particular","frequented","another","exist","where","matter","concerning","established","supposition","probable"]
    oov_serchable, oov_not_searchable = ngram_raprresentation(oov_searched, os.listdir(ALPHABET_FOLDER))
    print(f"OOV words searchable {len(oov_serchable)}")
    print(f"OOV words NOT searchable {len(oov_not_searchable)}")
    print(oov_not_searchable)


    n_class = 0
    n_items = 0
    min_5 = 0
    max_item_perclass = 0
    min_item_perclass = float('inf')

    # Alphabet stats
    n_class, n_items, max_item_perclass, min_item_perclass, min_5 = count_apphabet_items()

    #GT stats
    gt_set = count_words(GT_TRAINING_FOLDER)
    test_set = count_words(GT_FOLDER)

    iv_words_list = gt_set.keys() & test_set.keys()
    OOV_words_list = test_set.keys() - gt_set.keys()

    iv_words = { curr_key: test_set[curr_key] for curr_key in iv_words_list}
    oov_words = { curr_key: test_set[curr_key] for curr_key in OOV_words_list}

    oov_serchable, oov_not_searchable = ngram_raprresentation(OOV_words_list, os.listdir(ALPHABET_FOLDER))

    save_on_file(gt_set, os.path.join(OUT_FOLDER,"stats/gt_set.txt"))
    save_on_file(test_set, os.path.join(OUT_FOLDER,"stats/test_set.txt"))
    save_on_file(oov_words, os.path.join(OUT_FOLDER,"stats/oov.txt"))
    save_on_file(iv_words, os.path.join(OUT_FOLDER,"stats/iv.txt"))
    save_on_file(oov_serchable, os.path.join(OUT_FOLDER,"stats/oov_serchable.txt"))
    save_on_file(oov_not_searchable, os.path.join(OUT_FOLDER,"stats/oov_not_serchable.txt"))

    print(f"ALPHABET PATH:{ALPHABET_FOLDER}\n")
    print(f"The training set contains {len(gt_set)} different words")
    print(f"The test set contains {len(test_set)} different words")
    print(f"IV words {len(iv_words)}")
    print(f"OOV words {len(oov_words)}")
    print(f"OOV words searchable {len(oov_serchable)}")
    print(f"OOV words NOT searchable {len(oov_not_searchable)}")
    print()
    print(f"n_class: {n_class}\nn_items: {n_items}\nmax_percalss: {max_item_perclass}\nmin_perclass: {min_item_perclass}\nclass less3 item: {min_5}")