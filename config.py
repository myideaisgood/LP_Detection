import argparse

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    # Design paramters
    parser.add_argument('--img_size', type=int, default=448)             
    parser.add_argument('--iou_threshold', type=float, default=0.7)
    parser.add_argument('--conf_threshold', type=float, default=0.35)
    parser.add_argument('--nms_threshold', type=float, default=0.5)

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay_step', type=int, default=10)
    parser.add_argument('--decay_rate', type=int, default=0.1)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)                        
    parser.add_argument('--eval_every', type=int, default=1)                        

    # Directory parameters
    parser.add_argument('--data_dir', type=str, default="../DATASET/CCPD2019/")
    parser.add_argument('--dataset', type=str, default="ccpd/")
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
