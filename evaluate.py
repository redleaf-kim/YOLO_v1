import os
import numpy as np
import cv2

from collections import defaultdict
from model import YOLO_v1, VOC_CLASS_BGR
from model import visualize_boxes


def average_precision(recall, precision):
    """ Compute AP for one class.
        Args:
            recall: (numpy array) recall values of precision-recall curve.
            precision: (numpy array) precision values of precision-recall curve.
        Returns:
            (float) average precision (AP) for the class.
        """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = 0.0  # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


def evaluate(preds, targets, cls_names, threshold=0.5):
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        cls_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    """
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    aps = [] # list of average precisions (APs) for each class.

    for cls_name in cls_names:
        cls_preds = preds[cls_name] # all predicted objects for this class.

        if len(cls_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---class {} AP {}---'.format(cls_name, ap))
            aps.append(ap)
            break

        image_fnames = [pred[0]  for pred in cls_preds]
        probs        = [pred[1]  for pred in cls_preds]
        boxes        = [pred[2:] for pred in cls_preds]

        # Sort lists by probs.
        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes        = [boxes[i]        for i in sorted_idxs]

        # Compute total number of ground-truth boxes. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == cls_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box) in enumerate(zip(image_fnames, boxes)):

            if (filename, cls_name) in targets:
                boxes_gt = targets[(filename, cls_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if iou >= threshold:
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, cls_name)] # remove empty element from the dictionary.
                        break
            else:
                pass # this detection is FP.

        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = average_precision(recall, precision)
        print('---class {} AP {}---'.format(cls_name, ap))
        aps.append(ap)

    # Compute mAP by averaging APs for all classes.
    print('---mAP {}---'.format(np.mean(aps)))
    return aps



if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    from torchvision.models import vgg16_bn

    image_dir = './data/VOC2012'
    label_dir = './data/voc2012_val.txt'


    cls_names = list(VOC_CLASS_BGR.keys())
    targets = defaultdict(list)
    preds   = defaultdict(list)

    with open(label_dir, 'r') as f:
        lines = f.readlines()

    annotations = list()
    for line in lines:
        annotation = line.strip().split()
        annotations.append(annotation)

    image_names = []
    for annotation in annotations:
        fname = annotation[0]
        image_names.append(fname)

        num_bboxes = (len(annotation) -1)//5
        for b in range(num_bboxes):
            x1 = int(annotation[5 * b + 1])
            y1 = int(annotation[5 * b + 2])
            x2 = int(annotation[5 * b + 3])
            y2 = int(annotation[5 * b + 4])

            cls_label = int(annotation[5*b + 5])
            cls_name = cls_names[cls_label]

            targets[(fname, cls_name)].append([x1, y1, x2, y2])

    feature_extractor = vgg16_bn().features
    yolo_model = YOLO_v1(nms_thresh=0.5, prob_thresh=0.1, _model='vgg', _device='cuda').cuda()
    yolo_model.load_state_dict(torch.load('./yolo_v1.ptr'))

    yolo_model.eval()
    for fname in tqdm(image_names):
        path = os.path.join(image_dir, fname)
        image = cv2.imread(path)

        boxes, class_names, probs = yolo_model.detect(image)
        plot_img = visualize_boxes(image, boxes, class_names, probs)

        saveD = os.path.join('./debug', 'test')
        os.makedirs(saveD, exist_ok=True)
        saveD = os.path.join(saveD, '{:s}'.format('voc2007'))
        os.makedirs(saveD, exist_ok=True)
        cv2.imwrite(os.path.join(saveD, '{:s}.jpg'.format(str(fname))), plot_img)


        for box, class_name, prob in zip(boxes, class_names, probs):
            (centre_x, centre_y), (width, height) = box
            x1, x2 = int(centre_x - 0.5 * width), int(centre_x + 0.5 * width)
            y1, y2 = int(centre_y - 0.5 * height), int(centre_y + 0.5 * height)

            preds[class_name].append([fname, prob, x1, y1, x2, y2])

    print('Evaluate the detection result...')
    evaluate(preds, targets, cls_names=cls_names)