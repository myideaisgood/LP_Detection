import torch
from torch.utils.data import DataLoader

import os
import logging
import numpy as np
from tqdm import tqdm

from config import parse_args
from utils.dataloader import CCPD_Dataset
from utils.helpers import *
from models.yolo_net import Yolo

import cv2

from utils.helpers import get_bestpred
from utils.helpers import calc_iou

def eval(args, network, test_dataloader, device, logging):

    IOU_THRESHOLD = args.iou_threshold
    CONF_THRESHOLD = args.conf_threshold
    NMS_THRESHOLD = args.nms_threshold
    TEMP_DIR = 'temp_result/'
    os.makedirs(TEMP_DIR, exist_ok=True)

    total_sample = len(test_dataloader)
    correct_sample = 0

    with torch.no_grad():
        network.eval()

        for batch_idx, data in enumerate(tqdm(test_dataloader)):

            img, label = data

            img = img.to(device)
            label = label.numpy()

            # To network
            logits = network(img)
            predictions = post_processing(logits, args.img_size, ['LP'], network.anchors, CONF_THRESHOLD, NMS_THRESHOLD)

            if len(predictions) != 0:
                predictions = predictions[0]

                best_prediction = get_bestpred(predictions)

                iou = calc_iou(best_prediction, label)

                if batch_idx < 10:

                    img = img.detach().cpu().numpy().astype(np.uint8)
                    img = np.transpose(img, (0,2,3,1))
                    img = img[0]

                    rect = best_prediction
                    lx, ly, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                    rx, ry = lx+w, ly+h

                    imgs = img.copy()
                    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(imgs, (lx, ly), (rx, ry), (255,0,0), 2)
                    cv2.imwrite(TEMP_DIR + str(batch_idx).zfill(2) +'.jpg', imgs)
            else:
                iou = 0

            if iou >= IOU_THRESHOLD:
                correct_sample +=1

        accuracy = correct_sample/total_sample

        logging.info('*****Evaluation : %.4f' % (accuracy))

    return accuracy


if __name__ == "__main__":
    args = parse_args()

    IMGSIZE = args.img_size
    IOU_THRESHOLD = args.iou_threshold

    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    DATA_DIR = args.data_dir
    DATASET = args.dataset
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
    BEST_WEIGHTS = args.best_weights

    # Set up logger
    filename = os.path.join(LOG_DIR, 'logs_eval.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    # Set up Dataset
    test_dataset = CCPD_Dataset(DATA_DIR, 'val', imgSize=IMGSIZE)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False
    )

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up Networks
    model = Yolo(num_classes=1)
    model.to(device)

    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, BEST_WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, BEST_WEIGHTS))
    model.load_state_dict(checkpoint['network'])
    logging.info('Recover completed.')

    eval(args, model, test_dataloader, device, logging)