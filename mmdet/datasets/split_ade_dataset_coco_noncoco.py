from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

#
# Split COCO dataset into VOC vs non-VOC
#


def split_dataset(dataset_file, inds_split1, split1_file, split2_file):
    print('processing dataset', dataset_file)

    with open(dataset_file) as f:
        dataset = json.load(f)
    categories = dataset['categories']
    inds_split2 = [i for i in range(len(categories)) if i not in inds_split1]

    categories_split1 = [categories[i] for i in inds_split1]
    categories_split2 = [categories[i] for i in inds_split2]
    cids_split1 = [c['id'] for c in categories_split1]
    cids_split2 = [c['id'] for c in categories_split2]
    print('Split 1: {} classes'.format(len(categories_split1)))
    for c in categories_split1:
        print('\t', c['name'])
    print('Split 2: {} classes'.format(len(categories_split2)))
    for c in categories_split2:
        print('\t', c['name'])

    annotations = dataset['annotations']
    annotations_split1 = []
    annotations_split2 = []
    for ann in annotations:
        if ann['category_id'] in cids_split1:
            annotations_split1.append(ann)
        elif ann['category_id'] in cids_split2:
            annotations_split2.append(ann)
        else:
            raise Exception('This should not happen')
    print('Split 1: {} anns; save to {}'.format(len(annotations_split1), split1_file))
    print('Split 2: {} anns; save to {}'.format(len(annotations_split2), split2_file))
    dataset_split1 = {
        'images': dataset['images'],
        'annotations': annotations_split1,
        'categories': dataset['categories']}
    dataset_split2 = {
        'images': dataset['images'],
        'annotations': annotations_split2,
        'categories': dataset['categories']}
    with open(split1_file, 'w') as f:
        json.dump(dataset_split1, f)
    with open(split2_file, 'w') as f:
        json.dump(dataset_split2, f)


# class indices of VOC classes in COCO (0-indexed in range(0, 80) classes)
# class indices of COCO classes in ADE20K (1-indexed in range(1, 150) classes)
voc_inds = (8, 13, 16, 20, 21, 24, 48, 51, 66, 68, 70, 75, 77, 81, 84, 90, 91, 99, 117, 119, 125, 128,
136, 137, 148, 149)

dataset_prefix = '/home/g1007540910/mmdetection/data/ADE20k_cocostyle/annotations/'
split1_prefix = '/home/g1007540910/mmdetection/data/ADE20k_cocostyle/annotations/split_coco_'
split2_prefix = '/home/g1007540910/mmdetection/data/ADE20k_cocostyle/annotations/split_noncoco_'
suffix = (
    'train.json',
    'val.json')

for s in suffix:
    split_dataset(dataset_prefix + s, voc_inds, split1_prefix + s, split2_prefix + s)
