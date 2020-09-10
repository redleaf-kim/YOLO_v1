import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision
import cv2
import torchvision.transforms as tfms


VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}


CLASS_NAME_LIST =  list(VOC_CLASS_BGR.keys())

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()



class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)



class DarkNet(nn.Module):
    def __init__(self, feature_only=False, _bn=True, _init=True):
        super(DarkNet, self).__init__()

        self.feature_only = feature_only
        self.feature = self._build_feature_layer(_bn)
        if not self.feature_only:
            self.fc = self._build_fc_layer()

    def forward(self, x):
        x = self.feature(x)
        if not self.feature_only:
            x = self.fc(x)
        return x

    def _build_feature_layer(self, bn):
        def conv_block(in_f, out_f, k, s, p, bn, pool):
            layer = [nn.Conv2d(in_f, out_f, k, s, p)]
            if bn:
                layer.append(nn.BatchNorm2d(out_f))
            layer.append(nn.LeakyReLU(0.1, True))
            if pool:
                layer.append(nn.MaxPool2d(2))
            return layer

        feature_extrator = nn.Sequential(
            *conv_block(3, 64, 7, 2, 3, bn, True),
            *conv_block(64, 192, 3, 1, 1, bn, True),


            *conv_block(192, 128, 1, 1, 0, bn, False),
            *conv_block(128, 256, 3, 1, 1, bn, False),
            *conv_block(256, 256, 3, 1, 1, bn, False),
            *conv_block(256, 512, 3, 1, 1, bn, True),


            *conv_block(512, 256, 1, 1, 0, bn, False),
            *conv_block(256, 512, 3, 1, 1, bn, False),
            *conv_block(512, 256, 1, 1, 0, bn, False),
            *conv_block(256, 512, 3, 1, 1, bn, False),
            *conv_block(512, 256, 1, 1, 0, bn, False),
            *conv_block(256, 512, 3, 1, 1, bn, False),
            *conv_block(512, 256, 1, 1, 0, bn, False),
            *conv_block(256, 512, 3, 1, 1, bn, False),
            *conv_block(512, 512, 1, 1, 0, bn, False),
            *conv_block(512, 1024, 3, 1, 1, bn, True),


            *conv_block(1024, 512, 1, 1, 0, bn, False),
            *conv_block(512, 1024, 3, 1, 1, bn, False),
            *conv_block(1024, 512, 1, 1, 0, bn, False),
            *conv_block(512, 1024, 3, 1, 1, bn, False),
        )
        return feature_extrator

    def _build_fc_layer(self):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 1000)
        )
        return fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class YOLO_v1(nn.Module):
    def __init__(
            self,
            f_size=7, n_bb=2, n_cls=20,
            conf_thresh=0.1, prob_thresh=0.2, nms_thresh=0.7,
            class_name_list = None,
            bn=True, _model="vgg", _device='cpu'):

        super(YOLO_v1, self).__init__()
        self.S, self.B, self.C = f_size, n_bb, n_cls

        renet = False
        darknet = False
        if _model == "resnet":
            fe = torchvision.models.resnet50(pretrained=True)
            fe = list(fe.children())[:-3]
            fe = nn.Sequential(*fe)
            renet = True
            darknet = True
        elif _model == "vgg":
            # fe = list(torchvision.models.vgg16_bn(pretrained=True).features)[:30]
            fe = list(torchvision.models.vgg16_bn(pretrained=True).features)
            fe = nn.Sequential(*fe)
        elif _model == "darknet":
            fe = DarkNet(feature_only=True, _bn=bn, _init=True)
            darknet = True


        self.feature = fe.to(_device)
        self.conv    = self._build_conv_layer(darknet, renet)
        self.fc      = self._build_fc_layer(darknet)
        self.reshape = Reshape((-1, self.S, self.S, 5 * self.B + self.C))

        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

        self.class_name_list = class_name_list if (class_name_list is not None) else list(VOC_CLASS_BGR.keys())
        self.to_tensor = tfms.Compose([
                            tfms.ToTensor(),
                            # tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            tfms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])

    def forward(self, x):
        x = self.feature(x)
        x = self.conv(x)
        x = self.fc(x)
        x = self.reshape(x)
        return x



    def detect(self, image_bgr, image_size=448, valid=False):

        """ Detect objects from given image.

        Args:

            image_bgr: (numpy array) input image in BGR ids_sorted, sized [h, w, 3].

            image_size: (int) image width and height to which input image is resized.

        Returns:

            boxes_detected: (list of tuple) box corner list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.

            class_names_detected: (list of str) list of class name for each detected boxe.

            probs_detected: (list of float) list of probability(=confidence x class_score) for each detected box.

        """
        if not valid:
            h, w, _ = image_bgr.shape
            img = cv2.resize(image_bgr, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is trained with RGB images.
            img = self.to_tensor(img)  # [image_size, image_size, 3] -> [3, image_size, image_size]
            img = img.unsqueeze(0)  # [3, image_size, image_size] -> [1, 3, image_size, image_size]
            img = img.cuda()
        else:
            n, _, h, w = image_bgr.size()
            img = image_bgr


        with torch.no_grad():
            pred_tensor = self.forward(img)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0)

        # Get detected boxes_detected, labels, confidences, class-probs.
        boxes_normalized_all, class_labels_all, confidences_all, class_probs_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], [] # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_probs_masked = class_probs_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked * class_probs_masked)
            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_probs_masked[ids])

        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)

        # Postprocess for box, labels, probs.

        boxes_detected, class_names_detected, probs_detected = [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            centre_x, centre_y = w * box_normalized[0], h * box_normalized[1] # unnormalize centres with image size.
            width, height     = w * box_normalized[2], h * box_normalized[3] # unnormalize y with image height.

            #x1, x2 = centre_x - 0.5 * width, centre_x + 0.5 * width
            #y1, y2 = centre_y - 0.5 * height, centre_y + 0.5 * height
            boxes_detected.append(((centre_x, centre_y), (width, height)))

            class_label = int(class_label) # convert from LongTensor to int.
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob) # convert from Tensor to float.
            probs_detected.append(prob)

        return boxes_detected, class_names_detected, probs_detected



    def decode(self, preds):

        """ Decode tensor into box coordinates, class labels, and probs_detected.

        Args:

            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)

        Returns:

            boxes: (tensor) [[centre_x, centre_y, width, height]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].

            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].

            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].

            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].

        """

        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_probs = [], [], [], []

        cell_size = 1.0 / float(S)
        for i in range(S): # for x-dimension.
            for j in range(S): # for y-dimension.
                class_prob, class_label = torch.max(preds[j, i, 5*B:], 0)

                for b in range(B):
                    conf = preds[j, i, 5*b + 4]
                    prob = conf * class_prob
                    if float(prob) < self.prob_thresh:
                        continue

                    # box hase [offset_x, offset_y, w, h]
                    box = preds[j, i, 5*b : 5*b + 4]

                    base_xy = torch.tensor([i, j]) * cell_size
                    xy = box[:2] * cell_size + base_xy
                    wh = box[2:]

                    normalized_box = torch.zeros_like(box)
                    normalized_box[:2] = xy
                    normalized_box[2:] = wh

                    # Append result to the lists.
                    boxes.append(normalized_box)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_probs.append(class_prob)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_probs = torch.stack(class_probs, 0) # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_probs = torch.FloatTensor(0)

        return boxes, labels, confidences, class_probs


    def nms(self, boxes, scores):
        """ Apply non maximum supression.

        Args:

        Returns:

        """
        threshold = self.nms_thresh

        # x1 = boxes[:, 0] - 0.5 * boxes[:, 2] # [n,]
        # y1 = boxes[:, 1] - 0.5 * boxes[:, 3] # [n,]
        # x2 = boxes[:, 0] + 0.5 * boxes[:, 2] # [n,]
        # y2 = boxes[:, 1] + 0.5 * boxes[:, 3] # [n,]

        boxes_ = torch.zeros_like(boxes)
        boxes_[:, 0] = boxes[:, 0] - 0.5 * boxes[:, 2] # [n,]
        boxes_[:, 1] = boxes[:, 1] - 0.5 * boxes[:, 3] # [n,]
        boxes_[:, 2] = boxes[:, 0] + 0.5 * boxes[:, 2] # [n,]
        boxes_[:, 3] = boxes[:, 1] + 0.5 * boxes[:, 3] # [n,]

        return torchvision.ops.nms(boxes_, scores, self.nms_thresh)
        # areas = (x2 - x1) * (y2 - y1) # [n,]
        #
        # _, ids_sorted = scores.sort(0, descending=True) # [n,]
        # ids = []
        # while ids_sorted.numel() > 0:
        #     # Assume `ids_sorted` size is [m,] in the beginning of this iter.
        #
        #     i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        #     ids.append(i)
        #
        #     if ids_sorted.numel() == 1:
        #         break # If only one box is left (i.e., no box to supress), break.
        #
        #     inter_x1 = torch.max(x1[i], x1[ids_sorted[1:]])  # [m-1, ]
        #     inter_y1 = torch.max(y1[i], y1[ids_sorted[1:]])  # [m-1, ]
        #     inter_x2 = torch.min(x2[i], x2[ids_sorted[1:]])  # [m-1, ]
        #     inter_y2 = torch.min(y2[i], y2[ids_sorted[1:]])  # [m-1, ]
        #     inter_w = (inter_x2 - inter_x1).clamp(min=0)     # [m-1, ]
        #     inter_h = (inter_y2 - inter_y1).clamp(min=0)     # [m-1, ]
        #
        #     inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        #     unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
        #     ious = inters / unions # [m-1, ]
        #
        #     # Remove boxes whose IoU is higher than the threshold.
        #     ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        #     if ids_keep.numel() == 0:
        #         break # If no box left, break.
        #     ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        # return torch.LongTensor(ids)


    def _build_conv_layer(self, _darknet=False, _renet=False):
        in_out = 1024 if _darknet else 512

        layers = []
        if _renet:
            layers.append(
                nn.AdaptiveAvgPool2d((14, 14))
            )

        layers.extend([
            nn.Conv2d(in_out, in_out, 3, 1, 1),
            nn.BatchNorm2d(in_out),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(in_out, in_out, 3, 2, 1),
            nn.BatchNorm2d(in_out),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(in_out, in_out, 3, 1, 1),
            nn.BatchNorm2d(in_out),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(in_out, in_out, 3, 1, 1),
            nn.BatchNorm2d(in_out),
            nn.LeakyReLU(0.1, True)
        ])
        conv_layer = nn.Sequential(*layers)
        return conv_layer


    def _build_fc_layer(self, _darknet=False):
        in_out = 1024 if _darknet else 512
        S, B, C = self.S, self.B, self.C

        fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*in_out, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5, False),  # inplace=True option cause a weired RunTimeError... what the heck
            nn.Linear(4096, S*S*(5*B + C)),
            nn.Sigmoid()
        )

        return fc_layer


def visualize_boxes(image, boxes, class_names, probs, name_bgr_dict=None, line_thickness=3, valid=False):
    if name_bgr_dict is None:
        name_bgr_dict = VOC_CLASS_BGR

    image_boxes = image.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        if not valid:
            # Draw box on the image.
            centre, wah = box
            centre_x, centre_y = centre
            width, height = wah

            left, top = int(centre_x - 0.5*width), int(centre_y - 0.5*height)
            right, bottom = int(centre_x + 0.5*width), int(centre_y + 0.5*height)
            bgr = name_bgr_dict[class_name]
        else:
            left, top = box[0], box[1]
            right, bottom = box[2], box[3]

            class_name = CLASS_NAME_LIST[class_name]
            bgr = name_bgr_dict[class_name]

        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = "{:s}   {:.2f}".format(class_name, prob*100)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes



if __name__ == "__main__":
    from torchsummaryM import summary
    from torchvision.models import vgg16_bn

    inps = torch.ones(1, 3, 448, 448)
    yolo = YOLO_v1(_model='resnet')
    summary(yolo, inps)