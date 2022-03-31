import argparse
from random import choices

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    # Design paramters
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--batch_max_length', type=int, default=7, help='Max Length of Predicted Word')
    parser.add_argument('--pad_image', type=str2bool, default=False, help='Pad when resize')

    parser.add_argument('--Transformation', type=str, default='TPS', choices=['None', 'TPS'])
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', choices=['VGG, RCNN, ResNet'])
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', choices=['None', 'BiLSTM'])
    parser.add_argument('--Prediction', type=str, default='CTC', choices=['CTC', 'Attn'])

    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--img_color', type=str, default='Gray', choices=['Gray', 'RGB'])
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20000)

    parser.add_argument('--optim_type', type=str, default='Adam', choices=['Adam', 'Adadelta'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=5)

    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--train_acc_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1)

    # Directory parameters
    parser.add_argument('--data_dir', type=str, default="../DATASET/CCPD2019")
    parser.add_argument('--experiment_name', type=str, default='default/')
    parser.add_argument('--ckpt_dir', type=str, default="ckpt/")
    parser.add_argument('--log_dir', type=str, default="log/")
    parser.add_argument('--weights', type=str, default="ckpt.pth")
    parser.add_argument('--best_weights', type=str, default="ckpt_best.pth")

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
