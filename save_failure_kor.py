import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nltk import edit_distance
import shutil
import logging
import os
import numpy as np
import cv2

from config import parse_args
from utils.dataloader import CCPD_Recognition_Dataset
from utils.dataloader_kor import KorLP_Recognition_Dataset
from utils.datatransformer import AlignCollate
from model import Model
from utils.helpers import *
from utils.CTCConverter import *

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

def save_failure(network, test_dataloader, device, converter, BATCH_MAX_LENGTH):

    FAILURE = 'failure/'
    if os.path.exists(FAILURE):
        shutil.rmtree(FAILURE)
    os.mkdir(FAILURE)

    total_samples = 0
    correct_sample = 0
    avg_distance = 0

    with torch.no_grad():
        network.eval()

        for batch_idx, data in enumerate(tqdm(test_dataloader)):

            imgs, labels = data
            texts, lengths = converter.encode(labels, batch_max_length=BATCH_MAX_LENGTH, device=device)

            BATCH_SIZE = imgs.size(0)

            # Data to cuda
            imgs = imgs.to(device)
            
            preds = network(imgs, texts)

            preds_size = torch.IntTensor([preds.size(1)] * BATCH_SIZE)
            _, preds_index = preds.max(2)
            decoded = converter.decode(preds_index, preds_size)
            
            for idx in range(BATCH_SIZE):
                if labels[idx] == decoded[idx]:
                    correct_sample += 1
                else:
                    cur_img = imgs[idx].permute(1,2,0).detach().cpu().numpy()
                    cur_img = (((cur_img*0.5)+0.5)*255).astype(np.uint8)

                    decode_labels = decode_province(labels[idx], province, province_replace)
                    decode_pred = decode_province(decoded[idx], province, province_replace)

                    savename = FAILURE + decode_labels + '_' + decode_pred + '.jpg'

                    cv2.imwrite(savename, cur_img)

                avg_distance += edit_distance(labels[idx], decoded[idx])

                total_samples += 1
        
        total_accuracy = correct_sample / total_samples
        avg_distance /= total_samples

        return total_accuracy, correct_sample, total_samples, avg_distance

if __name__ == '__main__':
    args = parse_args()

    IMGH = args.imgH
    IMGW = args.imgW
    BATCH_MAX_LENGTH = args.batch_max_length
    PAD = args.pad_image

    IMG_COLOR = args.img_color
    if IMG_COLOR == 'Gray':
        args.input_channel = 1
    elif IMG_COLOR == 'RGB':
        args.input_channel = 3

    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    DATA_DIR = args.data_dir
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
    BEST_WEIGHTS = args.best_weights

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))


    # Set up Dataset
    converter = CTCLabelConverter(chars)
    args.num_class = len(converter.character)

    test_dataset = KorLP_Recognition_Dataset(DATA_DIR, 'Validation', IMG_COLOR)
    Collate = AlignCollate(IMGH, IMGW, PAD)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        collate_fn=Collate
    )            

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # Network declare
    network = Model(args, device)

    network = network.to(device)    

    checkpoint = torch.load(os.path.join(CKPT_DIR, BEST_WEIGHTS))
    network.load_state_dict(checkpoint['network'])

    test_acc, correct_sample, total_samples, avg_distance = save_failure(network, test_dataloader, device, converter, BATCH_MAX_LENGTH)
