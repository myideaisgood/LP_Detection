import os
import numpy as np 
import cv2
import sys
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
sys.path.append('.')
from utils.datatransformer import AlignCollate, ResizeNormalize
from config import parse_args

class Hikoo_Recog_TestDataset(Dataset):
    def __init__(self, data_dir, img_color, imgH, imgW):

        self.data_dir = data_dir
        self.img_color = img_color
        self.transform = ResizeNormalize((imgW, imgH))

        IMG_DIR = 'images/'
        LABEL_DIR = 'labels/'

        img_path = []

        img_dir = os.path.join(data_dir, IMG_DIR)

        for file in os.listdir(img_dir):
            if file.endswith('.jpg'):
                img_path.append(os.path.join(img_dir,file))
            
        self.img_path = sorted(img_path)

        self.label_path = []
        for img_name in self.img_path:

            img_name = img_name.replace(IMG_DIR, LABEL_DIR)

            if img_name.endswith('.jpg'):
                self.label_path.append(img_name.replace('.jpg', '.txt'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):

        if self.img_color == 'Gray':
            img = cv2.cvtColor(cv2.imread(self.img_path[index]), cv2.COLOR_BGR2GRAY)
            H, W = img.shape
        elif self.img_color == 'RGB':
            img = cv2.cvtColor(cv2.imread(self.img_path[index]), cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape


        with open(self.label_path[index]) as file:
            lines = [line.rstrip() for line in file]

            lp_label = []

            for line in lines:
                line = line.split(' ')

                cx, cy, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                
                xs = W * (cx - w/2)
                ys = H * (cy - h/2)
                xe = W * (cx + w/2)
                ye = H * (cy + h/2)

                bbox = np.array([int(xs), int(ys), int(xe), int(ye)])

                lp_label.append(bbox)

            lp_label = np.array(lp_label)

        # Crop Recognition
        lp_patch = []

        for lp_lab in lp_label:
            xs, ys, xe, ye = lp_lab[0], lp_lab[1], lp_lab[2], lp_lab[3]
            lp_pat = img[ys:ye,xs:xe]
            lp_pat_tensor = self.transform(lp_pat)
            lp_patch.append(lp_pat_tensor)

        if len(lp_patch) != 0:
            lp_patch = torch.stack(lp_patch, 0)

        return img, lp_label, lp_patch, self.img_path[index]

class ResizeNormalize(object):

    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


if __name__ == "__main__":

    args = parse_args()
    IMGH = args.imgH
    IMGW = args.imgW
    PAD = args.pad_image
    SAVE_DIR = 'playground/'
    DATA_DIR = '../DATASET/hikoo_demo/'
    BATCH_SIZE = 1
    SAVE_NUM = 100
    IMG_COLOR = 'RGB'

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = Hikoo_Recog_TestDataset(DATA_DIR, IMG_COLOR, IMGH, IMGW)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )    

    count = 0
    total_lp_num = 0

    for batch_idx, data in enumerate(tqdm(dataloader)):

        # if batch_idx == SAVE_NUM:
        #     sys.exit(1)

        img, lp_label, lp_imgs, img_name = data

        img = img[0].numpy()
        lp_label = lp_label[0].numpy()
        img_name = img_name[0].split('/')[-1]

        lp_num, _ = lp_label.shape

        for lp_idx in range(lp_num):
            cur_lp = lp_label[lp_idx]
            cv2.rectangle(img, (cur_lp[0], cur_lp[1]), (cur_lp[2], cur_lp[3]), color=(0,0,255), thickness=8)

        outname = os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '_' + img_name)
        
        if IMG_COLOR == 'RGB':
            cv2.imwrite(outname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(outname, img)