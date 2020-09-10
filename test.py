import os, cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tfms
import torchvision.datasets as dsets
import PIL
from PIL import Image

from model import *
from torchvision.models import vgg16_bn, vgg16

#
# feature_extractor = vgg16_bn().features
model = YOLO_v1(_model='vgg', nms_thresh=0.4, prob_thresh=0.25)
from_dict = torch.load('./yolo_v1.ptr')

model.load_state_dict((from_dict))
model.eval()
print("Model Ready")


cnt = 1
debug_dir = 'debug'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def video_test(main_path, video_name, save=False):
    cnt = 1

    video_path = os.path.join(main_path, video_name)
    cap = cv2.VideoCapture(video_path)

    while True:
        _, frame = cap.read()

        if not _:
            break

        boxes, class_names, probs = model.detect(frame)
        plot_img = visualize_boxes(frame, boxes, class_names, probs)

        saveD = os.path.join(debug_dir, './test/')
        os.makedirs(saveD, exist_ok=True)
        saveD = os.path.join(saveD, '{:s}'.format(video_name))
        os.makedirs(saveD, exist_ok=True)

        if save:
            cv2.imwrite(os.path.join(saveD, 'test_{:s}.jpg'.format(str(cnt).zfill(5))), plot_img)
            cnt += 1

        cv2.imshow("Test", plot_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()

def img_test(img_folder_path):
    cnt = 1
    path = img_folder_path

    img_list = os.listdir(path)
    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        img_name = img_name.split('.')[0]

        image = cv2.imread(img_path)
        boxes, class_names, probs = model.detect(image)

        plot_img = visualize_boxes(image, boxes, class_names, probs)

        saveD = os.path.join(debug_dir, 'test')
        os.makedirs(saveD, exist_ok=True)
        saveD = os.path.join(saveD, '{:s}'.format('img'))
        os.makedirs(saveD, exist_ok=True)
        cv2.imwrite(os.path.join(saveD, '{:s}.jpg'.format(str(img_name))), plot_img)

def cam_test(cam_num, save=False):
    cap = cv2.VideoCapture(cam_num)

    cnt = 1
    while True:
        _, frame = cap.read()

        if not _:
            break

        boxes, class_names, probs = model.detect(frame)
        plot_img = visualize_boxes(frame, boxes, class_names, probs)

        saveD = os.path.join(debug_dir, './test/')
        os.makedirs(saveD, exist_ok=True)
        saveD = os.path.join(saveD, 'cam_{:s}'.format(str(cam_num)))
        os.makedirs(saveD, exist_ok=True)

        if save:
            cv2.imwrite(os.path.join(saveD, 'test_{:s}.jpg'.format(str(cnt).zfill(5))), plot_img)
            cnt += 1

        cv2.imshow("Test", plot_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()

if __name__ == "__main__":
    # path = 'C:\\Users\H\Desktop\PennFudanPed\PNGImages'
    # path = 'C:\\Users\H\Desktop\Test'
    # img_test(path)

    main_path = 'C:\\Users\\H\\Desktop\\'
    video_name = 'test.mp4'
    video_test(main_path, video_name)

    # cam_test(0)

