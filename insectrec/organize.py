import numpy as np
import pandas as pd
import os, shutil
import glob
from natsort import natsorted
import cv2
from utils import clean_folder, get_plate_names, export_labels_2019, SAVE_DIR, read_plate
import sys
from tqdm import tqdm
import git
repo = git.Repo('.', search_parent_directories=True)
created_data_path = f'{repo.working_tree_dir}/insectrec/created_data'

# CREATING NECESSARY DIRECTORIES FOR THE PROJECT
path_annotations = f'{created_data_path}/annotations/'
path_images = f'{created_data_path}/images/'
path_voc_annotations = f'{created_data_path}/voc_annotations/'
path_impy_crops_export = f'{created_data_path}/impy_crops_export/'
path_weights = f'{created_data_path}/weights/'
path_logs = f'{created_data_path}/logs/'
for path in [path_annotations, path_images, path_voc_annotations, 
			path_impy_crops_export, path_weights, path_logs]:
	if not os.path.isdir(path):
		os.mkdir(path)	

yolo_to_voc = True # In the end of the script, yolo annotations get converted to voc
extract_boxes = True # Only works if above is true. Bounding boxes extracted and saved as images
clean = True # Deleting previous data created here (i.e. except of logs and weights)
if clean:
	print(f'Cleaning directories..')
	clean_folder(path_annotations)
	clean_folder(path_images)
	clean_folder(path_voc_annotations)
	os.system(f'rm -rf {path_impy_crops_export}*')

# Get name data from the sticky plates (their names)
year = '2019' #input("Choose year: \n")
BASE_DATA_DIR = f"/home/kalfasyan/data/images/sticky_plates/{year}"
assert year in ['2018','2019'], 'Wrong year given'
plates = get_plate_names(year, base_dir=BASE_DATA_DIR)

# Create classes.txt for yolo annotations 
# and a class_mapping.csv with the human readable labels
export_labels_2019(base_dir=SAVE_DIR)
class_map = pd.read_csv('created_data/class_mapping.csv')
assert len(class_map), "Couldn't read class mapping"
sub = class_map[['class', 'class_encoded']].drop_duplicates()
nan_code = sub[sub['class'].isnull()]['class_encoded'].iloc[0]

# Create a dataframe to save some statistics about the plates
# such as the number of nans and number of unique insects per plate
short_platenames = pd.Series(plates).apply(lambda x: x.split("/")[-1][:-4])
df_stats = pd.DataFrame(columns=['nr_nans','unique_insects','annotated'], index=short_platenames)
all_specs = []

annotated_plates, incomplete_plates = [], []
# Extra pixels around the image to crop
extra_pixels = 5

# Loop through all plates and nested loop through all insects in the plates
for p, platename in tqdm(enumerate(plates)):

	pname = platename.split('/')[-1][:-4] # defining the platename
	if 'empty' in pname:
		continue
	plate_img = read_plate(platename) # reading the plate image

	H,W,_ = plate_img.shape

	spec = pd.read_csv(plates[p][:-4] + '.txt', sep="\t") # reading the specifications of the plate

	if p == 0: # fetching column names (only once needed)
		columns = [ii for ii in spec.columns if ii.endswith('.1')]
		colextensions = ['index', 'name plate', 'R','G','B']
		columns.extend(colextensions) # adding extra columns

	spec = spec[columns]
	spec.dropna(axis=0, how='any', inplace=True) # cleaning up

	# REPLACING COMMAS WITH DOTS
	tmp = [spec[col].str.replace(',','.').astype(float) for col in columns if col not in colextensions]
	spec = pd.concat(tmp, axis=1) # changing type to float
	del tmp

	# ADDING YOLO AND HUMAN-READABLE ANNOTATION TO COLUMNS
	sub_map = class_map[class_map['platename'] == pname][['idx','class_encoded']].set_index('idx')
	sub_map2 = class_map[class_map['platename'] == pname][['idx','class']].set_index('idx')
	spec['yolo_class'] = sub_map
	spec['normal_class'] = sub_map2

	# REMOVING UNWANTED CLASSES 
	spec = spec[spec.normal_class != 'st'] # removing "stuk" class
	spec = spec[spec.normal_class != 'vuil'] # removing "vuil" class
	spec = spec[spec.normal_class.apply(lambda x: '+' not in str(x))]
	# SELECTING WANTED CLASSES
	spec = spec[spec.normal_class.isin(['m','v','bl','c','wmv','v(cy)'])]

	spec_nr_classes = spec['yolo_class'].unique().shape[0]
	condition1 = (spec_nr_classes >= 1)
	condition2 = True # (spec['yolo_class'].unique()[0] not in [nan_code, np.nan])
	condition3 = (spec['yolo_class'].isnull().sum() != spec['yolo_class'].shape[0])

	df_stats.loc[pname] = pd.Series({'nr_nans': spec[spec['yolo_class'] == nan_code].shape[0], 
										'unique_insects': spec['yolo_class'][spec['yolo_class'] != nan_code].unique().shape[0],
										'annotated': False})

	# finding the annotated plates - i.e the ones that don't have all nans in 'class'
	if condition1 and condition2 and condition3:
		print(f'Found annotated data: {condition1 and condition2} ----> COPYING IT')
		annotated_plates.append(platename)
		print(platename)

		spec['yolo_class'].fillna(0, inplace=True)
		spec['yolo_class'] = spec['yolo_class'].astype(int)
		spec['yolo_x'] = np.abs(spec['Bounding Rect Right.1'] - np.abs(spec['Bounding Rect Left.1'] - spec['Bounding Rect Right.1']) /2) / W
		spec['yolo_y'] = np.abs(spec['Bounding Rect Bottom.1'] - np.abs(spec['Bounding Rect Top.1'] - spec['Bounding Rect Bottom.1']) /2) / H
		spec['width'] = np.abs(spec['Bounding Rect Left.1'] - spec['Bounding Rect Right.1']) 
		spec['height'] = np.abs(spec['Bounding Rect Top.1'] - spec['Bounding Rect Bottom.1']) 
		# Making extracted boxes squares (to avoid distortions in future resizing)
		spec['yolo_width'] = pd.concat([spec['width'], spec['height']], axis=1).max(axis=1) / W + extra_pixels
		spec['yolo_height'] = pd.concat([spec['width'], spec['height']], axis=1).max(axis=1) / H + extra_pixels

		ann_full_new = os.path.join( path_annotations , f"{pname}.txt" )
		img_full_new = os.path.join( path_images , pname ) + '.jpg'

		# SAVING IMAGES
		if not os.path.isfile( img_full_new ):
			print(f"Copying {pname}")
			cv2.imwrite(img_full_new, plate_img)
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

	else:
		incomplete_plates.append(platename)

df_specs = pd.concat(all_specs, axis=0)

df_stats.to_csv(f'{created_data_path}/df_stats.csv')
df_specs.to_csv(f'{created_data_path}/df_specs.csv')

if yolo_to_voc:
	# CONVERTING LABELS FROM YOLO ANNOTATIONS (txt) TO VOC (xml)
	os.system("python yolo_to_voc.py")
	# EXTRACTING BOUNDING BOXES AS IMAGES
	if extract_boxes:
		import sys
		sys.path.insert(0, '..')
		from impy.ObjectDetectionDataset import ObjectDetectionDataset
		sticky = ObjectDetectionDataset(imagesDirectory=path_images, 
										annotationsDirectory=path_voc_annotations,
										databaseName='sticky_plates')
		sticky.saveBoundingBoxes(outputDirectory=path_impy_crops_export)