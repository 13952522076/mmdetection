import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

# A dataset for ADE20k dataset and COCO translation.
# We translate the ADE20k dataset to a MS COCO style.
# Hence, the dataset is following the coco.py
# Created by Xu Ma, July


@DATASETS.register_module()
class ADE20kCOCODataset(CustomDataset):

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    # ADE20K class name, referring:
    # https://github.com/CSAILVision/placeschallenge/tree/master/instancesegmentation
    CLASSES = ("bed", "windowpane", "cabinet", "person", "door", "table", "curtain",
               "chair", "car", "painting", "sofa", "shelf", "mirror", "armchair", "seat",
               "fence", "desk", "wardrobe", "lamp", "bathtub", "railing", "cushion", "box",
               "column", "signboard", "chest of drawers", "counter", "sink", "fireplace",
               "refrigerator", "stairs", "case", "pool table", "pillow", "screen door",
               "bookcase", "coffee table", "toilet", "flower", "book", "bench", "countertop",
               "stove", "palm", "kitchen island", "computer", "swivel chair", "boat",
               "arcade machine", "bus", "towel", "light", "truck", "chandelier", "awning",
               "streetlight", "booth", "television receiver", "airplane", "apparel",
               "pole", "bannister", "ottoman", "bottle", "van", "ship", "fountain",
               "washer", "plaything", "stool", "barrel", "basket", "bag", "minibike",
               "oven", "ball", "food", "step", "trade name", "microwave", "pot", "animal",
               "bicycle", "dishwasher", "screen", "sculpture", "hood", "sconce", "vase",
               "traffic light", "tray", "ashcan", "fan", "plate", "monitor", "bulletin board",
               "radiator", "glass", "clock", "flag")

    def _cocoid2adeid(self,cat_id):
        cat_mapping = {
            1: 4,
            2: 83,
            3: 9,
            4: 100,
            5: 59,
            6: 50,
            7: 100,
            8: 53,
            9: 48,
            10: 90,
            11: 100,
            # 12: 25,
            12: 100,
            13: 100,
            14: 41,
            # 15: 82,
            # 16: 82,
            # 17: 82,
            # 18: 82,
            # 19: 82,
            # 20: 82,
            # 21: 82,
            # 22: 82,
            # 23: 82,
            # 24: 82,
            15: 100,
            16: 100,
            17: 100,
            18: 100,
            19: 100,
            20: 100,
            21: 100,
            22: 100,
            23: 100,
            24: 100,
            # 25: 73,
            25: 100,
            26: 100,
            27: 100,
            28: 100,
            29: 100,
            30: 100,
            31: 100,
            32: 100,
            33: 76,
            34: 100,
            35: 100,
            36: 100,
            37: 100,
            38: 100,
            39: 100,
            40: 64,
            41: 100,
            42: 100,
            43: 100,
            44: 100,
            45: 100,
            46: 100,
            # 47: 77,
            # 48: 77,
            # 49: 77,
            # 50: 77,
            # 51: 77,
            # 52: 77,
            # 53: 77,
            # 54: 77,
            # 55: 77,
            # 56: 77,
            47: 100,
            48: 100,
            49: 100,
            50: 100,
            51: 100,
            52: 100,
            53: 100,
            54: 100,
            55: 100,
            56: 100,
            57: 8,
            # 58: 11,
            58: 100,
            59: 100,
            60: 1,
            61: 6,
            62: 38,
            63: 85,
            64: 46,
            65: 100,
            66: 100,
            67: 100,
            68: 100,
            69: 80,
            70: 75,
            71: 100,
            72: 28,
            73: 30,
            74: 100,
            75: 99,
            76: 89,
            77: 100,
            78: 100,
            79: 100,
            80: 100
        }
        return cat_mapping[cat_id]

    def detail_analysis(self,cocoEval,iou_thr=0.5):
        imageIds = cocoEval.params.imgIds
        # categoryIds = cocoEval.params.catIds
        all_unknown_undetected = 0
        all_unknown_detected = 0
        all_known_undetected = 0
        all_known_detected = 0
        all_known_2_unknown = 0
        all_unknown_2_known = 0
        all_detected_objects = 0
        all_labeled_objects = 0
        all_known_num = 0
        all_unknown_num = 0
        str = "img_id\t gt\t dt\t k\t uk\t k_d\t uk_d\t " \
              "k_ud\t uk_ud\t k_2_uk\t uk_2_k\t \n"

        for i in range(0,len(imageIds)):


            all_dt = [_ for cId in cocoEval.params.catIds for _ in cocoEval._dts[imageIds[i], cId]]
            all_gt = [_ for cId in cocoEval.params.catIds for _ in cocoEval._gts[imageIds[i], cId]]
            unknown_undetected = len([gt for gt in all_gt if gt['category_id'] == 100])
            unknown_num = unknown_undetected
            known_undetected = len([gt for gt in all_gt if gt['category_id'] != 100])
            known_num = known_undetected
            unknown_detected = 0
            known_detected = 0
            known_2_unknown = 0
            unknown_2_known = 0
            detected_objects = len(all_dt)
            labeled_objects = len(all_gt)
            if len(all_gt) !=0 and len(all_dt)!=0:
                for g in all_gt:
                    for d in all_dt:
                        iou = maskUtils.iou([d['segmentation']], [g['segmentation']], [0])[0][0]
                        g_category_id = g['category_id']
                        d_category_id = g['category_id']

                        if g_category_id == d_category_id and g_category_id == 100 and iou >= iou_thr:
                            unknown_detected = unknown_detected + 1
                            unknown_undetected = unknown_undetected -1
                            break
                        if g_category_id == d_category_id and g_category_id != 100 and iou >= iou_thr:
                            known_detected = known_detected + 1
                            known_undetected = known_undetected -1
                            break
                        if iou >= iou_thr:
                            if g_category_id==100 and d_category_id!=100:
                                unknown_2_known = unknown_2_known+1
                            if g_category_id != 100 and d_category_id == 100:
                                known_2_unknown = known_2_unknown+1






                        # if g_category_id==d_category_id and g_category_id==100:
                        #     if iou>=iou_thr:
                        #         unknown_detected = unknown_detected+1
                        #     else:
                        #         unknown_undetected = unknown_undetected+1
                        # if g_category_id == d_category_id and g_category_id != 100:
                        #     if iou>=iou_thr:
                        #         known_detected = known_detected+1
                        #     else:
                        #         known_undetected = known_undetected+1
                        # if g_category_id != d_category_id and g_category_id != 100 and iou>=iou_thr:
                        #     known_2_unknown = known_2_unknown+1
                        # if g_category_id != d_category_id and g_category_id == 100 and iou>=iou_thr:
                        #     unknown_2_known = unknown_2_known+1

            all_unknown_undetected = all_unknown_undetected + unknown_undetected
            all_unknown_detected = all_unknown_detected + unknown_detected
            all_known_undetected = all_known_undetected + known_undetected
            all_known_detected = all_known_detected + known_detected
            all_known_2_unknown = all_known_2_unknown + known_2_unknown
            all_unknown_2_known = all_unknown_2_known + unknown_2_known
            all_detected_objects = all_detected_objects + detected_objects
            all_labeled_objects = all_labeled_objects + labeled_objects



            str = str + "{}\t {}\t {}\t {}\t {}\t {}\t {}\t " \
                        "{}\t {}\t {}\t {}\t \n".format(
                imageIds[i],labeled_objects,detected_objects,known_num, unknown_num,known_detected,unknown_detected,
                known_undetected,unknown_undetected,known_2_unknown,unknown_2_known
            )




            # for j in range(0,len(categoryIds)):
            #     gt = cocoEval._gts[imageIds[i],categoryIds[j]]
            #     dt = cocoEval._dts[imageIds[i],categoryIds[j]]
            #     iou = cocoEval.computeIoU(i,j)

        str = str + "\n\n____________________________________all____________________________________________\n\n"
        str = str + "--\t {}\t {}\t {}\t {}\t {}\t {}\t " \
                    "{}\t {}\t {}\t {}\t \n".format(
            all_labeled_objects, all_detected_objects, all_known_num, all_unknown_num, all_known_detected, all_unknown_detected,
            all_known_undetected, all_unknown_undetected, all_known_2_unknown, all_unknown_2_known
        )
        print(str)


    def sup_sub_classes_analysis(self,cocoEval):
        imageIds = cocoEval.params.imgIds

        coco_animal_id = [15,16,17,18,19,20,21,22,23,24]
        for i in range(0, len(imageIds)):

            all_dt = [_ for cId in cocoEval.params.catIds for _ in cocoEval._dts[imageIds[i], cId]]
            all_gt = [_ for cId in cocoEval.params.catIds for _ in cocoEval._gts[imageIds[i], cId]]
            if len(all_gt) != 0:
                for g in all_gt:
                    iou = 0
                    detected_id = None
                    d_score = None
                    matched = False
                    g_category_id = g['category_id']
                    if g_category_id == 82:
                        if len(all_dt) != 0:
                            for d in all_dt:
                                iou = maskUtils.iou([d['segmentation']], [g['segmentation']], [0])[0][0]
                                d_category_id = d['category_id']
                                if iou > 0:
                                    d_score = str(d['score'])[:5]
                                    detected_id = d_category_id
                                    if d_category_id in coco_animal_id:
                                        matched = True
                                    break
                        print("Image: {}\t  gt: {}\t detected: {}\t IoU: {}\t score: {}\t match: {}\t".
                              format(imageIds[i], g['category_id'], iou, detected_id, d_score, matched))




    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    cat_id = self.cat_ids[label]
                    data['category_id'] = cat_id
                    # data['category_id'] = self._cocoid2adeid(cat_id)
                    # if data['category_id'] !=100:
                    #     continue # here we just focus on unkown objetcts
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style"""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    cat_id = self.cat_ids[label]
                    data['category_id'] = cat_id
                    # data['category_id'] = self._cocoid2adeid(cat_id)
                    # # if data['category_id'] !=100:
                    # #     continue # here we just focus on unkown objetcts
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    cat_id = self.cat_ids[label]
                    data['category_id'] = cat_id
                    # data['category_id'] = self._cocoid2adeid(cat_id)
                    # # if data['category_id'] !=100:
                    # #     continue # here we just focus on unkown objetcts
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float(f'{cocoEval.stats[i + 6]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                # Xu's detail analysis
                # self.detail_analysis(cocoEval, 0.5)
                self.sup_sub_classes_analysis(cocoEval)
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = f'{metric}_{metric_items[i]}'
                    val = float(f'{cocoEval.stats[i]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
