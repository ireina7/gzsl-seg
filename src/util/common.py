import argparse
from types import FunctionType
import torch
import torch.nn as nn
import numpy as np # type: ignore
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import matplotlib.pyplot as plt # type: ignore

from src.config import *
from util.typing.basic import *
from util.logging import logger


def get_arguments():
    """Parse all the arguments provided from the CLI.

    @Returns: A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    def arg(param: str, type=None, default=None, help=None, action=None):
        parser.add_argument(
            f'--{param}', type=type, default=default, help=help, action=action
        )
    #end arg
    
    arg("batch-size", type=int, default=BATCH_SIZE,
        help="Number of images sent to the network in one step."
    )
    arg("num-workers", type=int, default=NUM_WORKERS,
        help="number of workers for multithread dataloading."
    )
    arg("ignore-label", type=int, default=IGNORE_LABEL,
        help="The index of the label to ignore during the training."
    )
    arg("iter-size", type=int, default=ITER_SIZE,
        help="Accumulate gradients for ITER_SIZE iterations."
    )
    arg("input-size", type=str, default=INPUT_SIZE,
        help="Comma-separated string with height and width of source images."
    )
    # arg("is-training", action="store_true",
    #     help="Whether to updates the running means and variances during the training."
    # )
    arg("learning-rate", type=float, default=LEARNING_RATE,
        help="Base learning rate for training with polynomial decay."
    )
    arg("momentum", type=float, default=MOMENTUM,
        help="Momentum component of the optimiser."
    )
    # arg("not-restore-last", action="store_true",
    #     help="Whether to not restore last (FC) layers."
    # )
    arg("num-classes", type=int, default=NUM_CLASSES,
        help="Number of classes to predict (including background)."
    )
    arg("num-epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs."
    )
    arg("power", type=float, default=POWER,
        help="Decay parameter to compute the learning rate."
    )
    # arg("random-mirror", action="store_true",
    #     help="Whether to randomly mirror the inputs during the training."
    # )
    # arg("random-scale", action="store_true",
    #     help="Whether to randomly scale the inputs during the training."
    # )
    arg("random-seed", type=int, default=RANDOM_SEED,
        help="Random seed to have reproducible results."
    )
    arg("restore-from-where", type=str, default=RESTORE_FROM_WHERE,
        help="Where restore model parameters from pretrained or saved."
    )
    arg("save-num-images", type=int, default=SAVE_NUM_IMAGES,
        help="How many images to save."
    )
    arg("save-pred-every", type=int, default=SAVE_PRED_EVERY,
        help="Save summaries and checkpoint every often."
    )
    arg("snapshot-dir", type=str, default=SNAPSHOT_DIR,
        help="Where to save snapshots of the model."
    )
    arg("weight-decay", type=float, default=WEIGHT_DECAY,
        help="Regularisation parameter for L2-loss."
    )
#    parser.add_argument("--cpu", action="store_true", help="choose to use cpu device.")
#    parser.add_argument("--cpu", default=USE_CPU, help="choose to use cpu device.")
    arg("device", default=DEVICE,
        help="CPU or GPU"
    )
    # arg("tensorboard", action="store_true", 
    #     help="choose whether to use tensorboard."
    # )
    arg("log-dir", type=str, default=LOG_DIR,
        help="Path to the directory of log."
    )
    arg("dataroot", type=str, default=DATA_PATH,
        help="Path to the file listing the data."
    )
    arg("peek-if", type=doc('Func'), default=0,
        help="peek-if"
    )
    arg("eval-if", type=doc('Func'), default=0,
        help="eval-if"
    )
    arg("show-msg-if-epoch", type=doc('Func'), default=EPOCH_TO_SHOW_MSG,
        help="if current_epoch mod show-msg-every-epoch == 0 then can show msg"
    )
    arg("show-msg-if-loop", type=doc('Func'), default=LOOP_TO_SHOW_MSG,
        help="if current_loop mod show-msg-every-loop == 0 then can show msg"
    )
    arg("show-fig-if-epoch", type=doc('Func'), default=EPOCH_TO_SHOW_FIGURE,
        help="if current_epoch mod show-fig-every-epoch == 0 then can show figure"
    )
    arg("show-fig-if-loop", type=doc('Func'), default=LOOP_TO_SHOW_FIGURE,
        help="if current_loop mod show-fig-every-loop == 0 then can show figure"
    )
    return parser.parse_args()
#end parse



def lr_poly(base_lr, iter_, max_iter, power):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, num_steps, args, times=1):
    lr = lr_poly(args.learning_rate, i_iter, num_steps, args.power)
    optimizer.param_groups[0]["lr"] = lr * times




def bit_get(val, idx):
    """
    Gets the bit value.
    @Arg val: Input value, int or numpy int array.
    @Arg idx: Which bit of the input val.
    @Returns: The "idx"-th bit of input val.
    """
    return (val >> idx) & 1

def create_pascal_label_colormap(class_num):
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    @Returns: A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((class_num, 3), dtype=int)
    ind = np.arange(class_num, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


n_classes = 20
color = create_pascal_label_colormap(n_classes)

def to_color_img(img):
    #下面的0代表batch的第0个元素
    score_i = img[0,...]
    score_i = score_i.cpu().numpy()
    #转换通道
    score_i = np.transpose(score_i,(1,2,0))
    # one hot转一个channel
    score_i = np.argmax(score_i,axis=2)
    #color为上面生成的color list
    color_img = color[score_i]
    return color_img


"""
print_args : (args: Namespace) -> Void
"""
def print_args(args):
    dic = vars(args)
    for k in dic:
        logger.log("    {}: {}".format(k, dic[k]))
        #end for
    #end print_args

'''
print_config : (args: Namespace) -> Void
'''
def print_config(args):
    logger.log("Program configurations:")
    print_args(args)
    logger.log('End of configurations.')
    print()
    #end print_config


#def mIoU(pred, label):
def confusion_matrix(a, b, n):
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    #print(np.unique(a[k]), np.unique(b[k]))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))








class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # self.ce = nn.BCELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input, target):
        logp = self.ce(input, target)
        # print(f'logp:{logp}')
        p = torch.exp(-logp)
        # print(f'p:{p}')
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()




class _Mode(object):
    def __init__(self, mode: int):
        self.mode = mode

    def has(self, 
        mode_element#: seen | unseen | train | val
        ) -> bool:
        return (mode_element.mode & self.mode) != 0b0000
    #end class _Mode


class Mode(object):
    """
    Modes of Zero-Shot Segmentation.
    |seen | unseen | train | val|
    |higher  <-  bit  ->  lower |
    """
    seen   = _Mode(0b1000)
    unseen = _Mode(0b0100)
    train  = _Mode(0b0010)
    val    = _Mode(0b0001)

    def of(*modes) -> _Mode:
        res = 0
        for mode in modes:
            res = res | mode.mode
        return _Mode(res)

    train_seen        = of(train, seen)           # 1010
    train_unseen      = of(train, unseen)         # 0110
    train_seen_unseen = of(train, seen, unseen)   # 1011
    val_seen          = of(val, seen)             # 1001
    val_unseen        = of(val, unseen)           # 0101
    val_seen_unseen   = of(val, seen, unseen)     # 1101

    #end class Mode



