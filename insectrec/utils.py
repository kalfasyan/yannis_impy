import pandas as pd
import numpy as np
seed = 42
np.random.seed(42)
from natsort import natsorted
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

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
    elif year in ['2019','2020']:
        results_dir = []
        dir_to_look = os.path.join(base_dir, year)
        sticky_plate_date_dirs = [os.path.join(dir_to_look, f) for f in natsorted(os.listdir(dir_to_look))]
        for m in sticky_plate_date_dirs:
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

def export_labels(dict_or_df='df', years=None, created_data_dir=None):
    import os
    import glob
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    messed_up_index_plates = ["beauvech_w38_A_F10_51 mm_ISO160_1-15 s",
                            "beauvech_w39_A_F10_51 mm_ISO160_1-15 s",
                            "beauvech_w39_C_F10_51 mm_ISO160_1-15 s",
                            "brainelal_w38_A_F10_51 mm_ISO160_1-15 s",
                            "brainelal_w38_C_F10_51 mm_ISO160_1-15 s",
                            "brainelal_w38_B_F10_51 mm_ISO160_1-15 s",
                            "brainelal_w39_A_F10_51 mm_ISO160_1-15 s",
                            "brainelal_w39_B_F10_51 mm_ISO160_1-15 s",]

    dataframes = []
    for year in years:
        print(f"\n-- Processing expert labels for year: {year} --\n")
        labels_dir = f"{created_data_dir}/expert_labels/{year}"
        if not os.path.isdir(labels_dir):
            os.mkdir(labels_dir)

        xlsx_files = [fname for fname in glob.iglob(labels_dir + '/**/*.xlsx', recursive=True)]
        assert len(xlsx_files), "No expert labels found. (excel files provided by a Proefcentrum)"

        print(f'Number of excel annotation files found: {len(xlsx_files)}')
        wanted_columns_set = set(['name plate', 'index', 'Klasse', 'klasse'])
        df_labeldata = []
        for f in xlsx_files:
            print(f"Processing annotation file: {f.split('/')[-1]}")
            if f.endswith('w00.xlsx'):
                print(f"Skipping file: {f.split('/')[-1]}")
                continue
            
            sub = pd.read_excel(f)
            assert sub.iloc[:,1].name == 'index'
            assert sub.iloc[:,1].iloc[0] == 0., 'Check if excel file index starts with 1 instead of 0.'        
            df = sub[list(wanted_columns_set.intersection(sub.columns))]
            assert len(df.columns) == 3, 'Check excel file columns.'
            df.columns = map(str.lower, df.columns)
            df.rename(columns={'name plate': 'platename', 'klasse': 'class', 'index': 'idx'}, inplace=True)
            
            problematic_inds = []
            fixed_subdfs = []
            for i,q in df.groupby('platename'):

                if q.idx.iloc[0] != 0:
                    print(f"{q.idx.iloc[0]} found instead of 0 in first index of plate. Discarding it")
                    problematic_inds.append(q.iloc[0].name)

                if i in messed_up_index_plates:
                #    q.idx = q.idx + 1.
                   q['class'] = q['class'].shift(+1)
                fixed_subdfs.append(q)

            df = pd.concat(fixed_subdfs, axis=0)
            df.drop(problematic_inds, axis=0, inplace=True)
            assert df.idx.isna().sum() == 0
            df_labeldata.append(df)

        sub = pd.concat(df_labeldata, axis=0)
        sub['class'] = sub['class'].apply(lambda x: str(x).replace(" ","").lower())
        # sub['class'] = sub['class'].apply(lambda x: str(x).replace("2",""))
        # sub['class'] = sub['class'].apply(lambda x: str(x).replace("3",""))
        # sub['class'] = sub['class'].apply(lambda x: str(x).replace("4",""))
        dataframes.append(sub)

    df = pd.concat(dataframes, axis=0)
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'].tolist())

    path_annotations = f'{created_data_dir}/annotations/'
    assert os.path.isdir(path_annotations), "Annotations path not created."

    mapped = dict(zip(le.transform(le.classes_), le.classes_))
    # Saving class mapping to use as yolo annotation classes
    pd.Series(mapped).to_csv(f'{path_annotations}/classes.txt', sep=' ')
    # Saving class mapping to use when processing each plate
    df.to_csv(f'{created_data_dir}/class_mapping.csv')

    return None

def analyze_img_info():
    import os
    os.system(f"identify {SAVE_DIR}/images/*.JPG | grep 'x' > created_data_info.txt")
    df = pd.read_csv('created_data_info.txt', sep=' ', header=None)
    df['height'] = df[2].apply(lambda x: str(x).split('x')[0])
    df['width'] = df[2].apply(lambda x: str(x).split('x')[1])
    return df

def clean_folder(folder):
    """
    Function to clean a directory from all of its contents.
    """
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
    elif year in ['2019', '2020']:
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

def get_dataset(dataset='./insectrec/created_data/impy_crops_export/', img_dim=65, encode_labels=True):
    from imutils import paths
    from numpy import random
    import cv2
    from sklearn.model_selection import train_test_split    
    
    # initialize the data and labels
    print(" loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_dim, img_dim))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    if encode_labels:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.2, random_state=42)
    return trainX, testX, trainY, testY, labels

def train_generator(X_train, y_train, batch_size, nb_classes=9, img_dim=150):

    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)): 
                data = cv2.imread(train_batch[i])
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = cv2.resize(data, (img_dim, img_dim)) 
                data = img_to_array(data) / 255.0

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32) 
            y_batch = np.array(y_batch, np.float32)
            y_batch = to_categorical(y_batch, nb_classes)

            yield x_batch, y_batch


def valid_generator(X_val, y_val, batch_size, nb_classes=9, img_dim=150):

    while True:
        for start in range(0, len(X_val), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_val))
            train_batch = X_val[start:end]
            labels_batch = y_val[start:end]
            
            for i in range(len(train_batch)): 
                data = cv2.imread(train_batch[i])
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = cv2.resize(data, (img_dim, img_dim))
                data = img_to_array(data) / 255.0

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            y_batch = to_categorical(y_batch, nb_classes)

            yield x_batch, y_batch

def get_labelencoder_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

def augment_trainset(df_train=None,
                    augment_classes=[],
                    aug_imgs_path=None, img_dim=80, 
                    nb_batches=100, batch_size=512):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2
    import pandas as pd

    if len(augment_classes):
        print(f"df_train before selecting classes: {df_train.shape}")
        df_train = df_train[df_train['y_train'].isin(augment_classes)]
        print(f"df_train after selecting classes: {df_train.shape}")

    print(" Reading image data and assigning labels...")
    data = []

    imagePaths = df_train.X_train.tolist()
    np.random.seed(42)
    np.random.shuffle(imagePaths)
    labels = []
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_dim, img_dim))
        image = img_to_array(image)
        data.append(image)
        labels.append(imagePath.split('/')[-2])
    print('Normalizing data by dividing by 255.')
    data = np.array(data, dtype="float") / 255.0
    print('Creating an ImageDataGenerator.')
    aug = ImageDataGenerator(
                            rotation_range=60, 
                             width_shift_range=0.1,
                             height_shift_range=0.1, 
                            #  zoom_range=0.2, 
                             horizontal_flip=True, 
                             vertical_flip=True, 
                            #  brightness_range=[0.9,1.05],
    #                          zca_whitening=True,
                             fill_mode="nearest")
    print('Creating directories for each class.')
    rdm = np.random.randint(0,1e6)
    for i in np.unique(labels):
        if not os.path.isdir(f'{aug_imgs_path}/{i}'):
            os.mkdir(f'{aug_imgs_path}/{i}')
    print('Fitting data generator on data.')
    aug.fit(data)
    print('Expanding dataset by using generator\'s flow method on data/labels.')
    print(f'using batch_size of {batch_size}')
    batch_counter = 0
    for X_batch, y_batch in aug.flow(data, labels, batch_size=batch_size, seed=42):
        for i, mat in enumerate(X_batch):
            rdm = np.random.randint(0,1e6)
            img_savepath = f'{aug_imgs_path}/{y_batch[i]}/{y_batch[i]}_{rdm}{i}.jpg'
            img_mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_savepath, img_mat *255.)

        batch_counter += 1

        if batch_counter >= nb_batches:
            print(f"Finished augmentation in {nb_batches} batches of {batch_size}")
            break

def save_insect_crops(specifications, path_crops, plate_img):
    import cv2

    H,W,_ = plate_img.shape

    for _, row in specifications.iterrows():
        left  = int((row.yolo_x-row.yolo_width/2.)*W)
        right = int((row.yolo_x+row.yolo_width/2.)*W)
        top   = int((row.yolo_y-row.yolo_height/2.)*H)
        bot   = int((row.yolo_y+row.yolo_height/2.)*H)

        if(left < 0): left = 0;
        if(right > W-1): right = W-1;
        if(top < 0): top = 0;
        if(bot > H-1): bot = H-1;

        # print(f"left: {left}, right: {right}, top: {top}, bot: {bot}")
        crop = plate_img[top:bot, left:right]

        savepath = f"{path_crops}/{row.normal_class}/"
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        cv2.imwrite(f"{savepath}/{row.year}_{row.pname}_{row.insect_idx}.jpg", crop)

def to_weeknr(date=''):
    """
    Transforms a date strings YYYYMMDD to the corresponding week nr (e.g. 20200713 becomes w29)
    """
    week_nr = pd.to_datetime(date).to_pydatetime().isocalendar()[1]
    return f"w{week_nr}"