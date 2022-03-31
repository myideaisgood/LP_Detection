import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import os
import sys

sys.path.append('.')
from utils.datatransformer import *
from utils.CTCConverter import *

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

chars = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class CCPD_Recognition_Dataset(Dataset):
    def __init__(self, img_dir, subset, img_color):
        self.img_dir = img_dir
        self.img_color = img_color

        with open(os.path.join(img_dir, 'splits', subset) + '.txt') as file:
            self.img_paths = [line.rstrip() for line in file]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_paths[index])
        
        if self.img_color == 'Gray':
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
        elif self.img_color == 'RGB':
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        # Text Label
        label = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
        label = label.split('_')
        label = [int(i) for i in label]

        label = self.convert_label(label)

        # Bounding Box
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')

        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        lx, ly, rx, ry = leftUp[0], leftUp[1], rightDown[0], rightDown[1]

        # Crop Image
        if self.img_color == 'Gray':
            img = img[ly:ry, lx:rx]
        else:
            img = img[ly:ry, lx:rx, :]

        return (img, label)

    def convert_label(self, label):

        new_label = ""

        for idx, lbl in enumerate(label):
            if idx == 0:
                new_label += provinces[lbl]
            elif idx == 1:
                new_label += alphabets[lbl]
            else:
                new_label += ads[lbl]

        return new_label


if __name__ == "__main__":

    SAVE_DIR = 'playground/'
    DATA_DIR = '../DATASET/CCPD2019/'
    BATCH_SIZE = 4
    SAVE_NUM = 3
    IMG_COLOR = 'RGB'
    IMGH = 32
    IMGW = 100

    os.makedirs(SAVE_DIR, exist_ok=True)

    AlignCollate_ = AlignCollate(IMGH, IMGW, True)
    converter = CTCLabelConverter(chars)

    train_dataset = CCPD_Recognition_Dataset(DATA_DIR, 'train', IMG_COLOR)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        drop_last=True,
        collate_fn=AlignCollate_
    )

    for batch_idx, data in enumerate(train_dataloader):

        if batch_idx == SAVE_NUM:
            sys.exit(1)

        img, labels = data
        texts, lengths = converter.encode(labels, 8)

        for b_idx in range(BATCH_SIZE):
            cur_img = img[b_idx]
            cur_img = cur_img.permute(1,2,0).numpy()
            cur_img = (((cur_img * 0.5) + 0.5) * 255).astype(np.uint8)

            cur_label = labels[b_idx]
            cur_label_int = np.array(texts[b_idx])

            if IMG_COLOR == 'RGB':
                cv2.imwrite(os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '_' + str(b_idx) + '.jpg'), cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '_' + str(b_idx) + '.jpg'), cur_img)
            print(cur_label)
            out_string = ""
            for idx in cur_label_int:
                out_string += converter.character[idx]
            print(out_string)

        if batch_idx == SAVE_NUM - 1:
            sys.exit(1)