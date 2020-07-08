# Convert the png annotations to json file.
# Refer: https://github.com/CSAILVision/placeschallenge
# Created by Xu Ma.
# Date: July 04 2020

import os
import glob
import argparse
import json
import numpy as np
from scipy.misc import imread
from pycocotools import mask as COCOmask

padding=80
def cat_parsing(ori_id):
    mapping = {1: 60,
2: padding,
3: padding,
4: 1,
5: padding,
6: 61,
7: padding,
8: 57,
9: 3,
10: padding,
11: padding,
12: padding,
13: padding,
14: padding,
15: padding,
16: padding,
17: padding,
18: padding,
19: padding,
20: padding,
21: padding,
22: padding,
23: padding,
24: padding,
25: padding,
26: padding,
27: padding,
28: 72,
29: padding,
30: 73,
31: padding,
32: padding,
33: padding,
34: padding,
35: padding,
36: padding,
37: padding,
38: 62,
39: padding,
40: 74,
41: 14,
42: padding,
43: padding,
44: padding,
45: padding,
46: 64,
47: padding,
48: 9,
49: padding,
50: 6,
51: padding,
52: padding,
53: 8,
54: padding,
55: padding,
56: padding,
57: padding,
58: padding,
59: 5,
60: padding,
61: padding,
62: padding,
63: padding,
64: 40,
65: padding,
66: padding,
67: padding,
68: padding,
69: padding,
70: padding,
71: padding,
72: padding,
73: padding,
74: padding,
75: 70,
76: 33,
77: padding,
78: padding,
79: padding,
80: 69,
81: padding,
82: padding,
83: 2,
84: padding,
85: 63,
86: padding,
87: padding,
88: padding,
89: 76,
90: 10,
91: padding,
92: padding,
93: padding,
94: padding,
95: padding,
96: padding,
97: padding,
98: padding,
99: 75,
100: padding}
    return mapping[ori_id]



# all mapping classes
split_id_28classes = [4, 83, 9, 59, 50, 53, 48, 90, 25, 41, 82, 73,
            76, 64, 77, 8, 11, 1, 6, 38, 85, 46, 80, 75, 28, 30, 99, 89]
# strict mapping class
split_id_23classes = [4, 83, 9, 59, 50, 53, 48, 90, 41,
            76, 64, 8, 1, 6, 38, 85, 46, 80, 75, 28, 30, 99, 89]



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation demo')
    parser.add_argument('--ann_dir', default='annotations_instance/validation')  # CHANGE ACCORDINGLY
    parser.add_argument('--imgCatIdsFile', default='imgCatIds.json')
    parser.add_argument('--output_json', default='instance_validation_unknown_23classes.json')
    # parser.add_argument('--parsing_2coco', action='store_true', help='Parsing ADE20K cat_id to COCO id.')
    parser.add_argument('--parsing_unknown', default='True', action='store_true',
                        help='Parsing ADE20K cat_id to COCO known and unknown.')
    parser.add_argument('--split', action='store_true',help='Split Coco and non-COCO.')

    args = parser.parse_args()
    return args


def convert(args):
    data_dict = json.load(open(args.imgCatIdsFile, 'r'))
    img2id = {x['file_name']: x['id']
              for x in data_dict['images']}
    img2info = {x['file_name']: x
                for x in data_dict['images']}

    categories = data_dict['categories']
    images = []
    images_unique = set()
    annotations = []
    ann_id = 0
    # loop over annotation files
    files_ann = sorted(glob.glob(os.path.join(args.ann_dir, '*.png')))
    for i, file_ann in enumerate(files_ann):
        if i % 50 == 0:
            print('#files processed: {}'.format(i))

        file_name = os.path.basename(file_ann).replace('.png', '.jpg')
        img_id = img2id[file_name]
        if file_name not in images_unique:
            images_unique.add(file_name)
            images.append(img2info[file_name])

        ann_mask = imread(file_ann)
        Om = ann_mask[:, :, 0]
        Oi = ann_mask[:, :, 1]

        # loop over instances
        for instIdx in np.unique(Oi):
            if instIdx == 0:
                continue
            imask = (Oi == instIdx)
            cat_id = Om[imask][0]
            # if args.parsing_2coco:
            #     cat_id = cat_parsing(cat_id) # Mapping ADE20k instance cat_id to COCO, padding 80

            if args.parsing_unknown:
                if cat_id not in split_id_23classes:
                    cat_id = 100

            if args.split:
                # if cat_id ==padding:
                #     continue
                if cat_id not in split_id_23classes:
                    continue

            # RLE encoding
            rle = COCOmask.encode(np.asfortranarray(imask.astype(np.uint8)))
            bbox = COCOmask.toBbox(rle).tolist()
            ann = {}
            ann['id'] = ann_id
            ann_id += 1
            ann['image_id'] = img_id
            ann['segmentation'] = rle
            ann['category_id'] = int(cat_id)
            ann['iscrowd'] = 0
            ann['bbox'] = bbox
            ann['area'] = np.sum(imask)
            annotations.append(ann)

    # data_dict['annotations'] = annotations
    print('#files: {}, #instances: {}'.format(len(files_ann), len(annotations)))

    data_out = {'categories': categories, 'images': images, 'annotations': annotations}
    with open(args.output_json, 'w') as f:
        json.dump(data_out, f)


if __name__ == '__main__':
    args = parse_args()
    convert(args)
