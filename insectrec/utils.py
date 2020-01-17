import pandas as pd
import numpy as np
from natsort import natsorted
import os

SAVE_DIR = "./created_data/"
week_folders = ['Week 30 31 32', 'week 36 37 38', 'Week  33 34 35', 'Week 26  28 29 30']

def read_plate(platename):
	import cv2
	import os
	img = cv2.imread(platename, 3)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def results_dir(year, week_nr=None, base_dir=None, week_folders=week_folders):
	if year == '2018':
		assert isinstance(week_nr, str), "Wrong week number type."
		week = [ii for ii in week_folders if ii.endswith(str(week_nr))] # selecting folder corresponding to week_nr
		assert len(week), "No week folder ending with given number!"
		week = week[0]
		results = [ii for ii in os.listdir(os.path.join(base_dir, week)) if ii.endswith('_platen')][0] + '/results/' # getting matching results dir
		results_dir = os.path.join(base_dir, week, results) # full results directory to return
		assert len(results_dir), "No results dir found!"
		return results_dir 
	elif year == '2019':
		results_dir = []
		mydirs = [os.path.join(base_dir, f) for f in natsorted(os.listdir(base_dir))]
		for m in mydirs:
			for root,subdirs,files in os.walk(m):
				for d in subdirs:
					if d == 'results':
						results_dir.append("{}/{}".format(root,d))
		return results_dir

def get_labels(dict_or_df='df', base_dir=None):
	import glob
	import pandas as pd
	xlsx_files = [fname for fname in glob.iglob(base_dir + '/**/*.xlsx', recursive=True)]
	xlsx_classes = [fname for fname in xlsx_files if fname.split('/')[-1].split('.')[-2].endswith('geklasseerd')]
	dataframes = []
	for f in xlsx_classes:
		sub = pd.read_excel(f)
		sub['ID'] = sub['name plate'].map(str) + '_' + sub['index'].map(str) # '_' + sub['Klasse'].map(str) + 
		dataframes.append(sub[['ID','Klasse']])
	df = pd.concat(dataframes, axis=0)
	if dict_or_df == 'dict':
		return	pd.Series(df['Klasse'].values, index=df['ID']).to_dict()
	elif dict_or_df == 'df':
		return df
	else:
		raise ValueError('Didnt specify what to return correctly!')

def export_labels_2019(dict_or_df='df', base_dir=None):
	import os
	import glob
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder
	xlsx_files = [fname for fname in glob.iglob(base_dir + '/**/*.xlsx', recursive=True)]
	df_labeldata = []
	for f in xlsx_files:
		sub = pd.read_excel(f)
		sub = sub[['name plate', 'index', 'Klasse']]
		sub.rename(columns={'name plate': 'platename', 'Klasse': 'class', 'index': 'idx'}, inplace=True)
		df_labeldata.append(sub)
	df = pd.concat(df_labeldata, axis=0)
	df['class'] = df['class'].apply(lambda x: str(x).replace(" ","").lower())
	df['class'] = df['class'].apply(lambda x: str(x).replace("2",""))
	df['class'] = df['class'].apply(lambda x: str(x).replace("3",""))
	df['class'] = df['class'].apply(lambda x: str(x).replace("4",""))
	
	le = LabelEncoder()
	df['class_encoded'] = le.fit_transform(df['class'].tolist())

	path_annotations = f'{SAVE_DIR}/annotations/'
	if not os.path.isdir(path_annotations):
		os.mkdir(path_annotations)

	mapped = dict(zip(le.transform(le.classes_), le.classes_))
	# Saving class mapping to use as yolo annotation classes
	pd.Series(mapped).to_csv(f'{path_annotations}/classes.txt', sep=' ')
	# Saving class mapping to use when processing each plate
	df.to_csv(f'{base_dir}/class_mapping.csv')

	return None

def analyze_img_info():
	import os
	os.system(f"identify {SAVE_DIR}/images/*.JPG | grep 'x' > created_data_info.txt")
	df = pd.read_csv('created_data_info.txt', sep=' ', header=None)
	df['height'] = df[2].apply(lambda x: str(x).split('x')[0])
	df['width'] = df[2].apply(lambda x: str(x).split('x')[1])
	return df

def clean_folder(folder):
	import shutil
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_plate_names(year='2019', base_dir=None):
	import glob
	if year == '2018':
		raise NotImplementedError("Refactoring needed for working with 2018 data.")
		# try:
		# 	DATADIR = results_dir(year, week_nr=sys.argv[1], base_dir=base_dir) # given a week number, select the data
		# 	plates = natsorted(glob.glob(DATADIR+'*.JPG'))
		# 	specs = natsorted(glob.glob(DATADIR+'*.txt'))
		# except:
		# 	raise ValueError("No week number given (as sys.argv)")
	elif year == '2019':
		data_dirs = results_dir(year, base_dir=base_dir) # given a week number, select the data

		# Find plates for all image types known to exist in our dirs
		img_types = ('*.jpg','*.JPG','*.png') 
		all_plates = []
		for d in data_dirs:
			for filetype in img_types:
				all_plates.extend(glob.glob(d+'/'+filetype))
		plates = pd.Series(natsorted([plate for plate in all_plates if not plate[:-4].endswith('overlay')]))
		no_dupl_idx = pd.Series(plates).apply(lambda x: x.split('/')[-1][:-4]).drop_duplicates().index.values
		plates = plates.loc[no_dupl_idx].tolist()

		# # Find specifications text files
		# specs = []
		# for d in data_dirs:
		# 	specs.extend(glob.glob(d+'/'+'*.txt'))
		# specs = natsorted(specs)	
	else:
		raise ValueError("Wrong year given!")
	return plates