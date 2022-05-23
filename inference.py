import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nltk import edit_distance
import logging
import os
import cv2
from PIL import ImageFont, ImageDraw, Image
from time import time

from config import parse_args
from utils.dataloader_hikoo import Hikoo_Recog_TestDataset
from utils.dataloader_jeju import JEJU_LP_Recog_TestDataset
from utils.datatransformer import AlignCollate
from model import Model
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


def eval(network, test_dataloader, device, converter):

    SAVE_DIR = 'inference_result/'
    create_path(SAVE_DIR)

    fontpath = "gulim.ttc"
    font = ImageFont.truetype(fontpath, 20)

    with torch.no_grad():
        network.eval()

        for batch_idx, data in enumerate(tqdm(test_dataloader)):
            
            img, lp_label, lp_imgs, img_name = data
            img_total_num += 1
            
            if len(lp_imgs) != 0:
                img = img[0].numpy()
                lp_label = lp_label[0].numpy()
                lp_num, _ = lp_label.shape

                lp_imgs = lp_imgs[0]
                img_name = img_name[0].split('/')[-1]

                BATCH_SIZE = lp_imgs.size(0)

                # Data to cuda
                lp_imgs = lp_imgs.to(device)

                preds = network(lp_imgs, None)

                preds_size = torch.IntTensor([preds.size(1)] * BATCH_SIZE)
                _, preds_index = preds.max(2)

                decoded = converter.decode(preds_index, preds_size)

                for lp_idx in range(lp_num):
                    cur_lp = lp_label[lp_idx]
                    cv2.rectangle(img, (cur_lp[0], cur_lp[1]), (cur_lp[2], cur_lp[3]), color=(0,0,255), thickness=8)

                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)

                for lp_idx in range(lp_num):
                    cur_lp = lp_label[lp_idx]
                    decode_pred = decode_province(decoded[lp_idx], province, province_replace)

                    draw.text((cur_lp[0]-40, cur_lp[1]-40), decode_pred, font=font, fill=(0,0,255,0))
                    
                img = np.array(img_pil)

                outname = os.path.join(SAVE_DIR, img_name)

            if IMG_COLOR == 'RGB':
                cv2.imwrite(outname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(outname, img)

        return 


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
    BATCH_SIZE = 1
    NUM_WORKERS = args.num_workers

    DATA_DIR = args.data_dir
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

    dataset = Hikoo_Recog_TestDataset(DATA_DIR, IMG_COLOR, IMGH, IMGW)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )    

    converter = CTCLabelConverter(chars)
    args.num_class = len(converter.character)

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # Network declare
    network = Model(args, device)

    network = network.to(device)    

    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, BEST_WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, BEST_WEIGHTS))
    network.load_state_dict(checkpoint['network'])
    logging.info('Recover completed.')

    eval(network, dataloader, device, converter)