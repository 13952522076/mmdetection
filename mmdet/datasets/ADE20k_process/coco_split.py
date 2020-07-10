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

# strict mapping class
split_coco_id_24classes = [60, 1, 61, 57, 3, 72, 73, 62, 74, 14, 64, 9, 6, 8, 5,
                           40, 70, 33, 69, 2, 63, 76, 10, 75 ]



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation demo')
    parser.add_argument('--ann_file', default='/Users/melody/Downloads/instances_val2017.json')  # CHANGE ACCORDINGLY
    parser.add_argument('--output_overlap_json', default='/Users/melody/Downloads/instances_val2017_24classes.json')
    parser.add_argument('--output_rest__json', default='/Users/melody/Downloads/instances_val2017_76classes.json')
    # parser.add_argument('--parsing_2coco', action='store_true', help='Parsing ADE20K cat_id to COCO id.')
    args = parser.parse_args()
    return args


def convert(args):
    data_dict = json.load(open(args.ann_file, 'r'))
    images = data_dict['images']
    licenses = data_dict['licenses']
    info = data_dict['info']
    categories = data_dict['categories']
    annotations = data_dict['annotations']
    print('#Images: {}, # totally instances: {}'.format(len(images), len(annotations)))

    overlap_ann = []
    rest_ann = []
    for i in range(0,len(annotations)):
        if i % 100 == 0:
            print('#files processed: {}'.format(i))
        if annotations[i]['category_id']in split_coco_id_24classes:
            overlap_ann.append(annotations[i])
        else:
            rest_ann.append(annotations[i])

    overlap_out = {'licenses': licenses,
                   'categories': categories,
                   'images': images,
                   'annotations': overlap_ann,
                   'info': info
                   }
    rest_out = {'licenses': licenses,
                   'categories': categories,
                   'images': images,
                   'annotations': rest_ann,
                   'info': info
                   }
    print("{}: instance: {}".format(args.output_overlap_json, len(overlap_ann)))
    with open(args.output_overlap_json, 'w') as f:
        json.dump(overlap_out, f)
    print("{}: instance: {}".format(args.output_rest__json, len(rest_ann)))
    with open(args.output_rest__json, 'w') as f:
        json.dump(rest_out, f)


if __name__ == '__main__':
    args = parse_args()
    convert(args)
