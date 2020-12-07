import numpy as np
seed = 42
np.random.seed(seed)
import pandas as pd
import os, shutil, glob, cv2, sys, argparse
from natsort import natsorted
from utils import clean_folder, get_plate_names, export_labels, SAVE_DIR, read_plate, save_insect_crops
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
path_annotations = f'{created_data_path}/annotations/'
path_images = f'{created_data_path}/images/'
path_voc_annotations = f'{created_data_path}/voc_annotations/'
path_crops_export = f'{created_data_path}/crops_export/'
path_images_augmented = f'{created_data_path}/images_augmented/'
path_weights = f'{created_data_path}/weights/'
path_logs = f'{created_data_path}/logs/'
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
	os.system(f'rm {created_data_path}/df_*')
	os.system(f'rm {created_data_path}/class_mapping.csv')
assert len(os.listdir(path_crops_export)) <= 0, "Wrong"

# Get name data from the sticky plates (their names)
BASE_DATA_DIR = f"{args.datadir}"
years = args.years
assert all([y in ['2019','2020'] for y in years]), 'Wrong year given or in wrong format.'
plates = []
for y in years:
	y_plates = get_plate_names(y, base_dir=BASE_DATA_DIR)
	plates += y_plates
	print(f"Number of plates: {len(y_plates)} for year: {y}")

	# Create classes.txt for yolo annotations 
	# and a class_mapping.csv with the human readable labels

print(f"Number of ALL plates: {len(plates)}")

export_labels(created_data_dir=created_data_path, years=years)
class_map = pd.read_csv(f'{created_data_path}/class_mapping.csv')
assert len(class_map), "Couldn't read class mapping"
sub = class_map[['class', 'class_encoded']].drop_duplicates()
nan_code = sub[sub['class'].isnull()]['class_encoded'].iloc[0]

# Create a dataframe to save some statistics about the plates
# such as the number of nans and number of unique insects per plate
short_platenames = pd.Series(plates).apply(lambda x: x.split("/")[-1][:-4])
df_stats = pd.DataFrame(columns=['nr_nans','unique_insects','annotated'], index=short_platenames)
all_specs = []

annotated_plates, incomplete_plates = [], []

# Plates to ignore, since they were found to contain bad data (blurred/misclassified etc.)
bad_plates = ["beauvech_w38_B_F10_51 mm_ISO160_1-15 s",
            "brainlal_w27_A_58_160_1-15 s_11_48 mm_Manual_Manual_6240 x 4160",
            "kampen_w36_B_F10_51 mm_ISO160_1-15 s",
            "brainelal_8719_B_81_160_1-15 s_11_48 mm_Manual_Manual_6240 x 4160"]

labview_cols = ['Center of Mass X.1', 'Center of Mass Y.1', 'Bounding Rect Left.1',
       'Bounding Rect Top.1', 'Bounding Rect Right.1',
       'Bounding Rect Bottom.1', 'Equivalent Ellipse Major Axis.1',
       'Equivalent Ellipse Minor Axis.1', 'Area.1', 'Convex Hull Area.1',
       'Orientation.1', 'Ratio of Equivalent Ellipse Axes.1',
       'Ratio of Equivalent Rect Sides.1', 'Elongation Factor.1',
       'Compactness Factor.1', 'Heywood Circularity Factor.1', 'Type Factor.1',
       'R', 'G', 'B']

# Defining wanted classes
if args.nb_classes == 3: # using only the Fly classes
    wanted_classes = ['v', 'wmv', 'v(cy)']
elif args.nb_classes == 6:
    wanted_classes = ['m','v','bl','c','wmv','v(cy)']
elif args.nb_classes == 9:
    wanted_classes = ['m','v','bl','c','wmv','v(cy)','bv','sw','t']
elif args.nb_classes == 21:
    wanted_classes = ['m','v','bl','c','wmv','v(cy)','bv','gaasvlieg',
                    'grv','k','kever','nl','psylloidea','sp','sst','sw',
                    't','vlieg','weg','wnv','wswl']
else:
    raise ValueError(f"Number of classes not accepted: {args.nb_classes} ")
#['m','v','bl','c','wmv','v(cy)','bv','sw','t']
# ['m','v','bl','c','wmv','v(cy)','bv','gaasvlieg','grv','k','kever','nl','psylloidea','sp','sst','sw','t','vlieg','weg','wnv','wswl']
print(f"\nInsect classes selected: {wanted_classes}\n")

# Loop through all plates and nested loop through all insects in the plates
for p, platename in tqdm(enumerate(plates)):
    # Defining the plate name, week and year
    pname = platename.split('/')[-1][:-4] 
    if 'empty' in pname:
        continue
    pweek = pname.split('_')[1]
    pyear = platename.split('/')[len(BASE_DATA_DIR.split('/'))]

    # Skip some plates that you define in bad_plates
    if pname in bad_plates:
        print(f"\nSKIPPING BAD PLATE: {pname}")
        continue

    # Reading the specifications of the plate
    spec = pd.read_csv(plates[p][:-4] + '.txt', sep="\t") 
    # Fetching column names (only needed once)
    if p == 0: 
        columns = [ii for ii in spec.columns if ii.endswith('.1')]
        colextensions = ['index', 'name plate', 'R','G','B']
        columns.extend(colextensions) # adding extra columns
    spec = spec[columns]
    spec.rename(columns={'index': 'insect_idx'}, inplace=True)
    spec.dropna(axis=0, how='any', inplace=True)

    # ADDING YOLO AND HUMAN-READABLE ANNOTATION TO COLUMNS
    cmap = class_map[class_map['platename'] == pname].drop_duplicates(subset='idx', keep='first')
    if not len(cmap):
        print(f"Class mapping is empty for {platename}\nSkipping..")
        continue
    sub_map = cmap[['idx','class_encoded']].set_index('idx')
    sub_map2 = cmap[['idx','class']].set_index('idx')
    spec['yolo_class'] = sub_map
    spec['normal_class'] = sub_map2

    # REMOVING UNWANTED CLASSES 
    spec = spec[spec.normal_class != 'st'] # removing "stuk" class
    spec = spec[spec.normal_class != 'vuil'] # removing "vuil" class
    spec = spec[spec.normal_class.apply(lambda x: '+' not in str(x))]

    # SELECTING WANTED CLASSES
    spec = spec[spec.normal_class.isin(wanted_classes)]

    # Replacing commas from labview columns with dots 
    # to process them as floats
    for col in labview_cols:
        if spec[col].dtype != 'float64':
            spec[col] = spec[col].str.replace(",",".").astype(float)

    spec_nr_classes = spec['yolo_class'].unique().shape[0]
    condition1 = (spec_nr_classes >= 0)
    condition2 = True # (spec['yolo_class'].unique()[0] not in [nan_code, np.nan])
    condition3 = (spec['yolo_class'].isnull().sum() != spec['yolo_class'].shape[0])

    df_stats.loc[pname] = pd.Series({'nr_nans': spec[spec['yolo_class'] == nan_code].shape[0], 
                                        'unique_insects': spec['yolo_class'][spec['yolo_class'] != nan_code].unique().shape[0],
                                        'annotated': False})
    print(condition1, condition2, condition3)
    # finding the annotated plates - i.e the ones that don't have all nans in 'class'
    if condition1 and condition2 and condition3:

        # Reading the plate image
        plate_img = read_plate(platename) 
        H,W,_ = plate_img.shape
        print(f"img shape: {plate_img.shape}")

        print(f'\nFound annotated data for plate: {condition1 and condition2} ----> Copying plate')
        annotated_plates.append(platename)
        print(f"Platename: {platename.split('/')[-1]}")
        spec['pname'] = pname
        spec['year'] = pyear  
        # Making extracted boxes squares (to avoid distortions in future resizing)
        spec['width'] = 150
        spec['height'] = 150

        # Creating specifications according to 'YOLO' format
        spec['yolo_class'].fillna(0, inplace=True)
        spec['yolo_class'] = spec['yolo_class'].astype(int)
        spec['yolo_x'] = np.abs(spec['Bounding Rect Right.1'] - np.abs(spec['Bounding Rect Left.1'] - spec['Bounding Rect Right.1']) /2) / W
        spec['yolo_y'] = np.abs(spec['Bounding Rect Bottom.1'] - np.abs(spec['Bounding Rect Top.1'] - spec['Bounding Rect Bottom.1']) /2) / H
        spec['yolo_width'] = pd.concat([spec['width'], spec['height']], axis=1).max(axis=1) / W 
        spec['yolo_height'] = pd.concat([spec['width'], spec['height']], axis=1).max(axis=1) / H

        ann_full_new = os.path.join( path_annotations , f"{pname}.txt" )
        img_full_new = os.path.join( path_images , pname ) + '.jpg'

        # SAVING IMAGES
        if not os.path.isfile( img_full_new ):
            cv2.imwrite(img_full_new, plate_img)
        else:
            raise ValueError("IMAGE ALREADY EXISTS")
        # SAVING ANNOTATIONS
        if not len(spec) and not os.path.isfile( ann_full_new ):
            print('Empty file', ann_full_new)
            break
        else:#if not os.path.isfile( ann_full_new ):
            spec[['yolo_class','yolo_x','yolo_y','yolo_width','yolo_height']].to_csv(ann_full_new, sep=' ', index=False, header=False)

        df_stats.loc[pname] = pd.Series({'nr_nans': spec[spec['yolo_class'] == nan_code].shape[0], 
                                            'unique_insects': spec['yolo_class'][spec['yolo_class'] != nan_code].unique().shape[0],
                                            'annotated': True})

        all_specs.append(spec)
        if args.save_extractions:
            save_insect_crops(spec, path_crops_export, plate_img)

    else:
        incomplete_plates.append(platename)

df_specs = pd.concat(all_specs, axis=0)

# SAVING DATAFRAMES WITH STATISTICS REGARDING THE PLATES AND BOUNDING BOXES
df_stats.to_csv(f'{created_data_path}/df_stats.csv')
df_specs.to_csv(f'{created_data_path}/df_specs.csv')
print(f"path images: {path_images}")
print(f"path voc annotations: {path_voc_annotations}")

if args.yolo_to_voc:
	# CONVERTING LABELS FROM YOLO ANNOTATIONS (txt) TO VOC (xml)
	os.system("python yolo_to_voc.py")