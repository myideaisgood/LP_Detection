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
from utils.helpers import *


province = ['대구서', '동대문', '미추홀', '서대문', '영등포', '인천서', '인천중',
                    '강남', '강서', '강원', '경기', '경남', '경북', '계양', '고양', '관악', '광명', '광주', '구로', '금천', '김포', '남동', 
                    '대구', '대전', '동작', '부천', '부평', '서울', '서초', '안산', '안양', '양천', '연수', '용산', '인천', '전남', '전북', 
                    '충남', '충북', '영']

province_replace = ['괅', '놝', '돩', '랅', '맑', '밝', '삵', '앍', '잙', '찱',
                    '괉', '놡', '돭', '랉', '맕', '밡', '삹', '앑', '잝', '찵',
                    '괋', '놣', '돯', '뢇', '맗', '밣', '삻', '앓', '잟', '찷',
                    '괇', '놟', '돫', '뢃', '맓', '밟', '삷', '앏', '잛', '찳']

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '고', '구', '나', '너', '노', '누', '다', '더', '도', '두', 
        '라', '러', '로', '루', '마', '머', '모', '무', '바', '배', '버', '보', '부', '사', '서', '소', '수', '시', '아', '어', '오', 
        '우', '육', '자', '저', '조', '주', '지', '차', '카', '타', '파', '하', '허', '호', '히']

chars = chars + province_replace

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
        
            if not '-' in f['value']:
                if not '미주홀' in f['value']:
                    if ' ' in f['value']:
                        fixed = f['value'].replace(' ','')
                        self.img_paths.append(f['imagePath'])
                        fixed = encode_province(fixed, province, province_replace)
                        self.labels.append(fixed)
                    else:
                        fixed = encode_province(f['value'], province, province_replace)
                        self.img_paths.append(f['imagePath'])
                        self.labels.append(fixed)

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
    DATA_DIR = '../DATASET/KorLP/'
    BATCH_SIZE = 100
    SAVE_NUM = 3
    IMG_COLOR = 'Gray'
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

    sample_num = 0
    for batch_idx, data in enumerate(train_dataloader):

        # if batch_idx == SAVE_NUM:
        #     sys.exit(1)

        img, labels = data
        texts, lengths = converter.encode(labels, 9)

        for b_idx in range(BATCH_SIZE):

            cur_label = labels[b_idx]
            cur_label_int = np.array(texts[b_idx])

            print(cur_label)
            out_string = ""
            for idx in cur_label_int:
                out_string += converter.character[idx]
            print(decode_province(out_string, province, province_replace))

            sample_num += 1
            # cur_img = img[b_idx]
            # cur_img = cur_img.permute(1,2,0).numpy()
            # cur_img = (((cur_img * 0.5) + 0.5) * 255).astype(np.uint8)

            # cur_label = labels[b_idx]
            # cur_label_int = np.array(texts[b_idx])

            # if IMG_COLOR == 'RGB':
            #     cv2.imwrite(os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '_' + str(b_idx) + '.jpg'), cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR))
            # else:
            #     cv2.imwrite(os.path.join(SAVE_DIR, str(batch_idx).zfill(3) + '_' + str(b_idx) + '.jpg'), cur_img)
            # print(cur_label)
            # out_string = ""
            # for idx in cur_label_int:
            #     out_string += converter.character[idx]
            # print(out_string)
    
    print(sample_num)