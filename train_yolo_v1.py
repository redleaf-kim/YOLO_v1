import math
import argparse

import torchvision
from torchvision.models import vgg16_bn, vgg16
from torchsummaryM import summary
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import *
from loss import YOLO_loss
from utils import *


torch.autograd.set_detect_anomaly(True)

parser  = argparse.ArgumentParser(description="Pytorch Yolo model Training")
parser.add_argument('--cuda', default=True)
parser.add_argument('--batch', default=4)
parser.add_argument('--img', default='./data/VOC2012')
parser.add_argument('--label', default='./data/VOC2012.txt')
parser.add_argument('--feature', default='vgg')
parser.add_argument('--summary', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                     help='weight decay (default: 5e-4)',)
parser.add_argument('--start-epoch', default=1)
parser.add_argument('--epochs', default=135)
parser.add_argument('--print-freq', default=10)
parser.add_argument('--debug', default=False)
parser.add_argument('--bn', default=True)

def main():
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Set train & valid dataset 
    dataset = VOC_Dataset(args.img, args.label, debug=args.debug)
    valid_ratio = .01
    dataset_size = len(dataset)
    valid_size = int(valid_ratio * dataset_size)

    train_set, valid_set = random_split(dataset, [dataset_size-valid_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1)
    

    # Define feature_extractor and Yolo model based on
    # [You Only Look Once: Unified, Real-Time Object Detection] https://arxiv.org/abs/1506.02640
    yolo_model = YOLO_v1(bn=args.bn, _model=args.feature, _device=device).to(device)

    # Print summary of the model if needed
    if args.summary:
        summary(yolo_model, torch.ones(1, 3, 448, 448).to(device))

    
    # Define criterion & optimizer based on 
    # [You Only Look Once: Unified, Real-Time Object Detection] https://arxiv.org/abs/1506.02640
    criterion = YOLO_loss(f_size=yolo_model.S, _gpu=True)
    optimizer = torch.optim.SGD(yolo_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    
    debug_dir = 'debug'
    os.makedirs(debug_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs+1):
        yolo_model.train()
        cnt = 1
        train_loss = 0
        train_batch = 0
        valid_loss = 0
        valid_batch = 0

        dataset._train = True
        for i, (inps, tars) in enumerate(train_loader):
            update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1), args=args)
            lr = get_lr(optimizer) 

            inps = inps.float().to(device)
            tars = tars.float().to(device)

            
            preds  = yolo_model(inps)           
            loss = criterion(preds, tars)

            batch_size = inps.size(0)
            loss_cur_iter = loss.item()
            train_loss += loss_cur_iter * batch_size
            train_batch += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

           
            print('Epoch [{:3d}/{:3d}]    Iter [{:5d}/{:5d}]    LR: {:.4f}'.format(
                        epoch, args.epochs, i, len(train_loader), 
                        lr,
            ), end='')

            print('        Loss: {:.4f}    Average Loss: {:.4f}'.format(
                loss.item(),
                train_loss / float(train_batch)
            ))

            if i % 200 == 0:
                yolo_model.eval()
                dataset._train = False

                for j, (inps, tars) in enumerate(valid_loader):
                    # Load data as a batch.
                    inps = inps.float().to(device)

                    # Forward to compute validation loss.
                    with torch.no_grad():
                        boxes, class_names, probs = yolo_model.detect(image_bgr=inps, image_size=448, valid=True)

                        img = (inps.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 127.5) + 127.5
                        plot_img = visualize_boxes(img, boxes, class_names, probs, valid=False)

                        saveD = os.path.join(debug_dir, 'train_valid', '{:s}'.format(str(epoch).zfill(3)))
                        os.makedirs(saveD, exist_ok=True)
                        saveD = os.path.join(saveD, '{:s}'.format(str(i).zfill(5)))
                        os.makedirs(saveD, exist_ok=True)

                        cv2.imwrite(os.path.join(saveD, 'debug_{:s}.jpg'.format(str(cnt).zfill(5))), plot_img)
                        cnt += 1
            
                yolo_model.train()
        torch.save(yolo_model.state_dict(), "./yolo_v1.ptr")
        

def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0, args=None):
    if epoch == 1:
        lr = args.lr + (args.lr * 10 - args.lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 2:
        lr = args.lr * 10
    elif epoch == 76:
        lr = args.lr * 0.1
    elif epoch == 106:
        lr = args.lr * 0.01
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    main()
