# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
import pandas as pd


data_dir = './created_data/'

ANNOTATIONS_DIR_PREFIX = f"{data_dir}annotations/"
IMAGES_DIR_PREFIX = f"{data_dir}images/"
DESTINATION_DIR = f"{data_dir}converted_labels/"

mapping = f'{data_dir}class_mapping.csv'
mapdict = pd.read_csv(mapping, index_col='Unnamed: 0')[['class','class_encoded']].drop_duplicates().set_index('class_encoded').to_dict()['class']

CLASS_MAPPING = {str(k):str(v) for k,v in mapdict.items()}

def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = IMAGES_DIR_PREFIX
    ET.SubElement(root, "path").text = os.getcwd() + '/' + "{}.jpg".format(file_prefix)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(round(voc_label[1])))
        ET.SubElement(bbox, "ymin").text = str(int(round(voc_label[2])))
        ET.SubElement(bbox, "xmax").text = str(int(round(voc_label[3])))
        ET.SubElement(bbox, "ymax").text = str(int(round(voc_label[4])))
    return root


def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(DESTINATION_DIR, file_prefix))


def read_file(file_path):
    file_prefix = file_path.split(".txt")[0]
    if file_prefix != 'classes':
        image_file_name = "{}.jpg".format(file_prefix)
        print(image_file_name)
        img = Image.open("{}/{}".format(IMAGES_DIR_PREFIX, image_file_name))
        w, h = img.size
        with open(ANNOTATIONS_DIR_PREFIX+file_path, 'r') as file:
            lines = file.readlines()
            voc_labels = []
            for line in lines:
                voc = []
                line = line.strip()
                data = line.split()
                voc.append(CLASS_MAPPING.get(data[0]))
                bbox_width = float(data[3]) * w
                bbox_height = float(data[4]) * h
                center_x = float(data[1]) * w
                center_y = float(data[2]) * h
                voc.append(center_x - (bbox_width / 2))
                voc.append(center_y - (bbox_height / 2))
                voc.append(center_x + (bbox_width / 2))
                voc.append(center_y + (bbox_height / 2))
                voc_labels.append(voc)
            create_file(file_prefix, w, h, voc_labels)
        print("Processing complete for file: {}".format(file_path))


if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)
for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
    if filename.endswith('txt'):
        read_file(filename)
    else:
        print("Skipping file: {}".format(filename))
