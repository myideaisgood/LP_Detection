import torch.optim as optim
from torch.utils.data import DataLoader

import os
import logging
import numpy as np
from tqdm import tqdm

from config import parse_args
from utils.dataloader import *
from utils.average_meter import AverageMeter
from eval import eval
from models.loss import YoloLoss
from models.yolo_net import Yolo

args = parse_args()

IMGSIZE = args.img_size
IOU_THRESHOLD = args.iou_threshold

GPU_NUM = args.gpu_num
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EPOCHS = args.epochs
LR = args.lr
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate
PRINT_EVERY = args.print_every
SAVE_EVERY = args.save_every
EVAL_EVERY = args.eval_every

DATA_DIR = args.data_dir
DATASET = args.dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
BEST_WEIGHTS = args.best_weights

os.makedirs('experiments/', exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'test'), exist_ok=True)

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

train_dataset = CCPD_Dataset(DATA_DIR, 'train', imgSize=IMGSIZE)
test_dataset = CCPD_Dataset(DATA_DIR, 'val', imgSize=IMGSIZE)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True
)

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

# Set up Criterion
criterion = YoloLoss(1, model.anchors, 32)

# Load the pretrained model if exists
init_epoch = 0
best_metric = 0.0

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    best_metric = checkpoint['best_metric']
    LR = checkpoint['lr']
    model.load_state_dict(checkpoint['network'])
    logging.info('Recover completed. Current epoch = #%d' % (init_epoch))


# Set up Optimizer
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

# Train
# Train Network
for epoch_idx in range(init_epoch+1, EPOCHS):
    model.train()
    scheduler.step()

    lossAver = AverageMeter()
    lossCoord = AverageMeter()
    lossConf = AverageMeter()
    lossCls = AverageMeter()

    for batch_idx, data in enumerate(tqdm(train_dataloader)):    

        img, label = data

        img = img.to(device)
        label = label.numpy()
        label_list = []
        for idx in range(len(label)):
            label_list.append(np.expand_dims(label[idx],axis=0))

        # To network
        logits = model(img)

        # Loss
        loss, loss_coord, loss_conf, loss_cls = criterion(logits, label_list)

        # Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Holders
        lossAver.update(loss.item())
        lossCoord.update(loss_coord.item())
        lossConf.update(loss_conf.item())
        lossCls.update(loss_cls.item())

    scheduler.step()

    if epoch_idx % PRINT_EVERY == 0:
        logging.info('[Epoch %d/%d] Loss = %.4f = %.4f (Coord) + %.4f (Conf) + %.4f (Cls)' % (epoch_idx, EPOCHS, lossAver.avg(), lossCoord.avg(), lossConf.avg(), lossCls.avg()))

    if epoch_idx % SAVE_EVERY == 0:

        output_path = os.path.join(CKPT_DIR, WEIGHTS)
        
        torch.save({
            'epoch_idx': epoch_idx,
            'lr': optimizer.param_groups[0]["lr"],
            'best_metric': best_metric,
            'network': model.state_dict()
            }, output_path)

        logging.info('Model Saved')

    if epoch_idx % EVAL_EVERY == 0:

        accuracy = eval(args, model, test_dataloader, device, logging)

        # Save if result is the best
        if accuracy > best_metric:

            best_metric = accuracy

            output_path = os.path.join(CKPT_DIR, BEST_WEIGHTS)
            
            torch.save({
                'epoch_idx': epoch_idx,
                'lr': optimizer.param_groups[0]["lr"],
                'best_metric': best_metric,
                'network': model.state_dict()
                }, output_path)

            logging.info('Model Saved')                

        logging.info('*** Best Metric = %.4f' % (best_metric))