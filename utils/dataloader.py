import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import os
import sys

class CCPD_Dataset(Dataset):
    def __init__(self, img_dir, subset, imgSize):
        self.img_dir = img_dir

        with open(os.path.join(img_dir, 'splits', subset) + '.txt') as file:
            self.img_paths = [line.rstrip() for line in file]

        self.img_size = imgSize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_paths[index])
        img = cv2.imread(img_name)

        resizedImage = cv2.resize(img, (self.img_size, self.img_size))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')

        assert img.shape[0] == 1160
        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        lx, ly, rx, ry = leftUp[0], leftUp[1], rightDown[0], rightDown[1]
        lx, rx = lx*self.img_size/ori_w, rx*self.img_size/ori_w
        ly, ry = ly*self.img_size/ori_h, ry*self.img_size/ori_h
        
        new_w, new_h = rx -lx, ry - ly

        new_labels = np.array([lx, ly, new_w, new_h, 0])

        resizedImage = resizedImage.astype('float32')
        resizedImage = torch.from_numpy(resizedImage)
        resizedImage = resizedImage.permute(2,0,1)

        return resizedImage, new_labels    


if __name__ == "__main__":

    SAVE_DIR = 'playground/'
    DATA_DIR = '../DATASET/CCPD2019/'
    BATCH_SIZE = 4
    SAVE_NUM = 10
    imgSize = 448

    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset = CCPD_Dataset(DATA_DIR, 'train', imgSize)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        drop_last=True
    )

    for batch_idx, data in enumerate(train_dataloader):

        if batch_idx == SAVE_NUM:
            sys.exit(1)

        img, label = data

        img = (img).numpy().astype(np.uint8)
        img = np.transpose(img, (0,2,3,1))

        for b in range(BATCH_SIZE):
            cur_img = img[b].copy()
            
            lx = label[b][0].numpy()
            ly = label[b][1].numpy()
            w = label[b][2].numpy()
            h = label[b][3].numpy()

            lu = (int(lx), int(ly))
            rd = (int(lx+w), int(ly+h))

            cv2.rectangle(cur_img, lu, rd, (255,0,0), 2)

            if b == 0:
                imgs = cur_img
            else:
                imgs = np.concatenate([imgs, cur_img], axis=1)

        cv2.imwrite(os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '.jpg'), imgs)