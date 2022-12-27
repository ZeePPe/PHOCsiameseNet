import os
from tqdm import tqdm

GT_FOLDER = "data/GT"

for doc_folder in os.listdir(GT_FOLDER):
    doc_folder_path = os.path.join(GT_FOLDER, doc_folder)

    for gt_file in os.listdir(doc_folder_path):
        if len(gt_file.split("_")) <= 1:
            new_name = "gt_" + gt_file.split(".")[0] + "_" + doc_folder + ".txt"

            os.rename(  os.path.join(doc_folder_path, gt_file),
                        os.path.join(doc_folder_path, new_name))

print("Done!")