from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json


def read_dataset(dataset_file):
    print('processing dataset', dataset_file)

    with open(dataset_file) as f:
        dataset = json.load(f)
    categories = dataset['categories']
    print("categories: {}".format(categories))



dataset_prefix = '/home/g1007540910/mmdetection/data/ADE20k_cocostyle/annotations/val.json'


read_dataset(dataset_prefix)


