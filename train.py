import torch
from torch.utils.data import DataLoader

import logging
import os
from tqdm import tqdm
from nltk import edit_distance

from config import parse_args
from model import Model
from evaluate import eval
from utils.average_meter import AverageMeter
from utils.dataloader import CCPD_Recognition_Dataset
from utils.datatransformer import AlignCollate
from utils.helpers import *
from utils.CTCConverter import *

chars = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Design Parameters
IMGH = args.imgH
IMGW = args.imgW
BATCH_MAX_LENGTH = args.batch_max_length
PAD = args.pad_image
IMG_COLOR = args.img_color
if IMG_COLOR == 'Gray':
    args.input_channel = 1
elif IMG_COLOR == 'RGB':
    args.input_channel = 3

# Session Parameters
GPU_NUM = args.gpu_num
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
N_EPOCHS = args.epochs

OPTIM_TYPE = args.optim_type
LR = args.lr
BETA1 = args.beta1
RHO = args.rho
EPS = args.eps
GRAD_CLIP = args.grad_clip

TRAIN_ACC_EVERY = args.train_acc_every
SAVE_EVERY = args.save_every
PRINT_EVERY = args.print_every
EVAL_EVERY = args.eval_every

# Directory Parameters
EXP_NAME = args.experiment_name
DATA_DIR = args.data_dir
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
BEST_WEIGHTS = args.best_weights

# Check if directory does not exist
create_path('experiments/')
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'test'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
converter = CTCLabelConverter(chars)
args.num_class = len(converter.character)

train_dataset = CCPD_Recognition_Dataset(DATA_DIR, 'train', IMG_COLOR)
test_dataset = CCPD_Recognition_Dataset(DATA_DIR, 'test', IMG_COLOR)

Collate = AlignCollate(IMGH, IMGW, PAD)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True,
    collate_fn=Collate
)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network declare
network = Model(args, device)

network = network.to(device)
network = initialize_model(network)
filtered_parameters = filter_parameter(network)

# Set up Loss Functions
criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

# Load the pretrained model if exists
init_epoch = 0
best_metric = 0
best_distance = 1000

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    best_metric = checkpoint['best_metric']
    best_distance = checkpoint['best_distance']
    LR = checkpoint['lr']
    network.load_state_dict(checkpoint['network'])
    logging.info('Recover completed. Current epoch = #%d' % (init_epoch))

# Create Optimizer
if OPTIM_TYPE == 'Adam':
    optimizer = torch.optim.Adam(filtered_parameters, lr=LR, betas=(BETA1, 0.999))
elif OPTIM_TYPE == 'Adadelta':
    optimizer = torch.optim.Adadelta(filtered_parameters, lr=LR, rho=RHO, eps=EPS)

for epoch_idx in range(init_epoch+1, N_EPOCHS):

    # Metric holders
    losses = AverageMeter()

    # Network to train mode
    network.train()

    total_train_samples = 0
    train_correct_samples = 0
    train_avg_distance = 0

    # Train for batches
    for batch_idx, data in enumerate(tqdm(train_dataloader)):
                    
        imgs, labels = data
        texts, lengths = converter.encode(labels, batch_max_length=BATCH_MAX_LENGTH, device=device)

        # Data to cuda
        imgs = imgs.to(device)
        
        preds = network(imgs, texts)

        preds_size = torch.IntTensor([preds.size(1)] * BATCH_SIZE)
        preds_softmax = preds.log_softmax(2).permute(1, 0, 2)
        loss = criterion(preds_softmax, texts, preds_size, lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        losses.update(loss.item())

        if epoch_idx % TRAIN_ACC_EVERY == 0:
            SAMPLE_NUM = imgs.size(0)
            _, preds_index = preds.max(2)
            decoded = converter.decode(preds_index, preds_size)

            for idx in range(SAMPLE_NUM):
                if labels[idx] == decoded[idx]:
                    train_correct_samples += 1
                total_train_samples += 1

                train_avg_distance += edit_distance(labels[idx], decoded[idx])           

    # Print Epoch Measures
    if epoch_idx % PRINT_EVERY == 0:
        if epoch_idx % TRAIN_ACC_EVERY == 0:
            train_acc = train_correct_samples / total_train_samples
            train_avg_distance /= total_train_samples
            logging.info('[Epoch %d/%d] Loss = %.4f   Train Accuracy : %.1f [%d/%d]   Train Distance : %.1f' % (epoch_idx, N_EPOCHS, losses.avg(), train_acc*100, train_correct_samples, total_train_samples, train_avg_distance))
        else:
            logging.info('[Epoch %d/%d] Loss = %.4f ' % (epoch_idx, N_EPOCHS, losses.avg()))

    if epoch_idx % EVAL_EVERY == 0:

        test_acc, correct_sample, total_samples, avg_distance = eval(network, test_dataloader, device, converter, BATCH_MAX_LENGTH)

        logging.info('====== Evaluation Accuracy : %.1f  [%d/%d]   Edit Distance : %.1f' % (test_acc*100, correct_sample, total_samples, avg_distance))

        if test_acc > best_metric:

            best_metric = test_acc
            best_distance = avg_distance
            output_path = os.path.join(CKPT_DIR, BEST_WEIGHTS)
            
            torch.save({
                'best_metric': best_metric,
                'best_distance' : best_distance,
                'network': network.state_dict()
                }, output_path)

            logging.info('Best Model Saved')

        logging.info('====== Best Accuracy = %.1f   Best Distance = %.1f' % (best_metric*100, best_distance))

    if epoch_idx % SAVE_EVERY == 0:

        output_path = os.path.join(CKPT_DIR, WEIGHTS)

        torch.save({
            'epoch_idx': epoch_idx,
            'lr': optimizer.param_groups[0]["lr"],
            'best_metric': best_metric,
            'best_distance': best_distance,
            'network': network.state_dict()
            }, output_path)

        logging.info('Model Saved')