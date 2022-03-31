import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import os
import sys
import json

sys.path.append('.')
from utils.datatransformer import *
from utils.CTCConverter import *

provinces = ['강원', '경기', '경기고양나', '경기고양다', '경기고양사', '경기고양타', '경기광명파', '경기김포자', '경기부천다', '경기부천바', 
            '경기부천사', '경기부천아', '경기부천자', '경기부천차', '경기부천카', '경기부천타', '경기부천파', '경기부천하', '경기안산라', '경기안양바', 
            '경기인천아', '경남', '경북', '광주', '대구', '대구서하', '대전', '서울', '서울강남타', '서울강남파', '서울강남하', '서울강서자', 
            '서울강서차', '서울관악카', '서울관악파', '서울구로가', '서울구로자', '서울구로카', '서울금천자', '서울동대문파', '서울동작자', '서울서대문다', 
            '서울서대문마', '서울서대문바', '서울서초마', '서울양천마', '서울양천자', '서울양천차', '서울영등포아', '서울용산자', '영경기', '영서울', 
            '영인천', '인천', '인천계양사', '인천남동파', '인천미주홀파', '인천미추홀파', '인천부평사', '인천부평아', '인천서타', '인천연수나', 
            '인천연수다', '인천중마', '전남', '전북', '충남', '충북']

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '고', '구', '나', '너', '노', '누', '다', '더', '도', '두', '라', '러', 
        '로', '루', '마', '머', '모', '무', '바', '버', '보', '부', '사', '서', '소', '수', '시', '아', '어', '오', '우', '육', '자', '저', '조', 
        '주', '하', '허', '호', '히']


class KorLP_Recognition_Dataset(Dataset):
    def __init__(self, img_dir, subset, img_color):

        assert (subset == 'Training') or (subset == 'Validation')

        self.img_dir = img_dir
        self.img_color = img_color
        self.subset = subset

        # Read Label path
        label_path = []

        for file in os.listdir(os.path.join(img_dir, subset, 'label')):
            if file.endswith('.json'):
                label_path.append(file)

        label_path = sorted(label_path)

        self.img_paths = []
        self.labels = []

        for l_path in label_path:
            cur_path = os.path.join(img_dir, subset, 'label', l_path)
            f = json.load(open(cur_path))
        
            isProvince = bool([ele for ele in provinces if(ele in f['value'])])

            if not isProvince:
                if not '-' in f['value']:
                    if ' ' in f['value']:
                        fixed = f['value'].replace(' ','')
                        self.img_paths.append(f['imagePath'])
                        self.labels.append(fixed)
                    else:
                        self.img_paths.append(f['imagePath'])
                        self.labels.append(f['value'])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.subset, 'image', self.img_paths[index])
        
        if self.img_color == 'Gray':
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
        elif self.img_color == 'RGB':
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        # Text Label
        label = self.labels[index]

        return (img, label)


if __name__ == "__main__":

    SAVE_DIR = 'playground/'
    DATA_DIR = '../DATASET/KorLP_Small/'
    BATCH_SIZE = 4
    SAVE_NUM = 3
    IMG_COLOR = 'RGB'
    IMGH = 32
    IMGW = 100

    os.makedirs(SAVE_DIR, exist_ok=True)

    AlignCollate_ = AlignCollate(IMGH, IMGW, False)
    converter = CTCLabelConverter(chars)

    train_dataset = KorLP_Recognition_Dataset(DATA_DIR, 'Training', IMG_COLOR)

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