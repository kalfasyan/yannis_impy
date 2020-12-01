import numpy as np
seed = 42
np.random.seed(seed)
import pandas as pd
import os, shutil, glob, cv2, sys, argparse
from natsort import natsorted
from utils import clean_folder, export_labels, SAVE_DIR, read_plate, save_insect_crops
from utils_photobox import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', help="directory of sticky plate images")
parser.add_argument('--years', nargs='+')
parser.add_argument('--clean', dest='clean', action='store_true')
parser.add_argument('--no-clean', dest='clean', action='store_false')
parser.add_argument('--yolo_to_voc', dest='yolo_to_voc', action='store_true', 
                    help='In the end of the script, yolo annotations get converted to voc')
parser.add_argument('--save_extractions', dest='save_extractions', action='store_true', 
                    help='Whether to save the extracted insect crops ')
parser.add_argument('--nb_classes', type=int, choices=[3,6,9,21], default=6)

parser.set_defaults(clean=True, yolo_to_voc=True, save_extractions=True)

args = parser.parse_args()
assert isinstance(args.datadir, str) and os.path.isdir(args.datadir), 'Provide a valid path'
if not len(args.datadir): # /home/kalfasyan/data/images/sticky_plates/
    raise ValueError("Please provide a datadir argument.")

created_data_path = f'{args.datadir}/created_data'

# CREATING NECESSARY DIRECTORIES FOR THE PROJECT
path_annotations = f'{created_data_path}/annotations_photobox/'
path_images = f'{created_data_path}/images_photobox/'
path_voc_annotations = f'{created_data_path}/voc_annotations_photobox/'
path_crops_export = f'{created_data_path}/crops_export_photobox/'
path_images_augmented = f'{created_data_path}/images_augmented_photobox/'
path_weights = f'{created_data_path}/weights_photobox/'
path_logs = f'{created_data_path}/logs_photobox/'
for path in [created_data_path, path_annotations, path_images, path_voc_annotations, 
            path_crops_export, path_weights, path_logs, path_images_augmented]:
    if not os.path.isdir(path):
        os.mkdir(path)	

if args.clean:
    print(f'Cleaning directories..')
    clean_folder(path_annotations)
    clean_folder(path_images)
    clean_folder(path_voc_annotations)
    os.system(f'rm -rf {path_crops_export}*')
    os.system(f'rm -rf {path_images_augmented}*')
    os.system(f'rm {created_data_path}/df_photobox_*')
    # os.system(f'rm {created_data_path}/class_mapping_photobox.csv')
assert len(os.listdir(path_crops_export)) <= 0, "Wrong"

# Get name data from the sticky plates (their names)
BASE_DATA_DIR = f"{args.datadir}"
years = args.years
assert all([y in ['2020'] for y in years]), 'Wrong year given or in wrong format.'
plates = []
for y in years:
    print(y)
    y_plates = get_plate_names_photobox(y, base_dir=BASE_DATA_DIR)
    plates += y_plates
    print(f"Number of plates: {len(y_plates)} for year: {y}")

    # Create classes.txt for yolo annotations 
    # and a class_mapping.csv with the human readable labels

print(f"Number of ALL plates: {len(plates)}")

# Create a dataframe to save some statistics about the plates
# such as the number of nans and number of unique insects per plate
short_platepaths = pd.Series(plates).apply(lambda x: x.split("/")[-1][:-4])
df_stats = pd.DataFrame(columns=['nr_nans','unique_insects','annotated'], index=short_platepaths)
all_specs = []

annotated_plates, incomplete_plates = [], []

# Plates to ignore, since they were found to contain bad data (blurred/misclassified etc.)
bad_plates = []

print(f"Total number of plates : {len(plates)}")

all_dfs = []

# Loop through all plates and nested loop through all insects in the plates
for p, platepath in tqdm(enumerate(plates)):

    # Defining the platepath
    pname = platepath.split('/')[-1][:-4] 
    print(pname)

    # Skip some plates that you define in bad_plates
    if pname in bad_plates:
        print("SKIPPING BAD PLATE")
        continue

    # if pname != "herent_w35_4-30_4056x3040":
    #     continue

    resimg, df = overlay_image_nms(platepath, created_data_path, nms_threshold=0.08, plot_orig=True)
    all_dfs.append(df)


pd.concat(all_dfs).to_csv(f"{created_data_path}/results_asdf.xlsx", sep=',')