import os
import os.path as ospx
import torch

split = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
RESTORE_FROM_WHERE = "pretrained"
EMBEDDING = "all"
lambdaa = 0.2
#USE_CPU = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
NUM_WORKERS = 3
ITER_SIZE = 1
IGNORE_LABEL = 255 # the background
INPUT_SIZE = (512, 512)
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_EPOCHS = 50
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
WEIGHT_DECAY = 0.0005
LOG_DIR = "./log"
SHOW_EPOCH = 10
DIM_LATENT = 1024
weak_size = BATCH_SIZE
weak_proportion = 0.2

EPOCH_TO_SHOW_FIGURE = lambda epoch: epoch % 1 == 0
EPOCH_TO_SHOW_MSG    = lambda epoch: epoch % 1 == 0
LOOP_TO_SHOW_FIGURE  = lambda loop : loop % 100 == 0
LOOP_TO_SHOW_MSG     = lambda loop : loop % 1 == 0
EPOCH_TO_EVAL        = lambda epoch: epoch >= 5 or epoch % 10 == 0



DATA_PATH = "dataset"
PRETRAINED_OUR_PATH = "model/segmentation/pretrained/our_qfsl_confidence"
SNAPSHOT_PATH = "model/segmentation/snapshots/vgg/lambda_split_single_1"
PATH = "output"


DATA_VOC = ospx.join(DATA_PATH, 'voc')
DATA_SEM = ospx.join(DATA_PATH, 'semantic') # Semantic embeddings path
SNAPSHOT_DIR = ospx.join(PATH, SNAPSHOT_PATH, EMBEDDING)
RESULT_DIR = ospx.join(PATH, SNAPSHOT_PATH, "result.txt")


  
class path_singleton(object):
  """
  Only for creating a path singleton, very dirty and evil!
  Should never be called in user code.
  """
  def __init__(self):
    self.root = 'src'
    rooted = lambda p: ospx.join(self.root, p)
    self.dataset = rooted(DATA_PATH)
    self.dataset_voc = rooted(DATA_VOC)
    self.dataset_voc2012 = ospx.join(self.dataset_voc, 'voc2012')
    self.dataset_semantic = rooted(DATA_SEM)
    self.pretrained = rooted(PRETRAINED_OUR_PATH)
    self.snapshot = rooted(SNAPSHOT_PATH)
    self.output = rooted('output')

    self.join = ospx.join
#end path_singleton

path = path_singleton()
