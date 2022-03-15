import sys
import matplotlib.pyplot as plt

from util.typing.basic import *
from src.config import *


def show_figure_nonblocking() -> None:
  plt.show(block = False)
  plt.pause(0.001)
  #end show_figure_nonblocking

def show_figure_blocking() -> None:
  plt.show()
  #end show_figure_blocking

def show_if(EPOCH: int, LOOP: int):
  def f(epoch: int, loop: int, call_back) -> None:
    if epoch % EPOCH == 0 and loop % LOOP == 0:
      call_back()
  return f

show_figure_if = show_if(EPOCH_TO_SHOW_FIGURE, LOOP_TO_SHOW_FIGURE)
show_msg_if    = show_if(EPOCH_TO_SHOW_MSG,    LOOP_TO_SHOW_MSG   )


def draw_sample(batch):
  imgs, msks = batch['image'], batch['label']
  fig, axs = plt.subplots(1, 3, figsize=(10, 3))
  axs[0].imshow(imgs[0].permute(1, 2, 0))
  axs[1].imshow(msks[0], cmap = 'tab20', vmin = 0, vmax = 21)
  #end draw_sample

# def show_sample(batch):
#     draw_sample(batch)
#     log("Displaying image of {}".format(batch['name']))
#     # plt.colorbar()
#     show_figure_nonblocking()
#     #end show_sample


def show_single_figure_result(batch, pred, mask):
  ans = pred[0].clone().detach().cpu().numpy()
  #x = np.where(ans == 0, 255, ans)
  x = ans
  y = mask.cpu()[0]
  x[y == 255] = 255
  draw_sample(batch)
  # pyplot.figure()
  plt.imshow(x, cmap = 'tab20', vmin = 0, vmax = 21)
  plt.colorbar()
  # show_figure_nonblocking()







end = 0


default_files = {
  'msg': sys.stdout,
  'err': sys.stderr,
  'mod': './output',
}

class Logger(object):
  """
  The main Logger
  @param files: {
    msg: File | sys.stdout -- Message
    err: File | sys.stderr -- Error
    mod: File -- model
  } -- Determine where the logger should log.
  @param painter: Painter -- Determine how to log and show figures.
  """
  def __init__(self, files=default_files):
    self.files = files
    # self.painter = painter
  end
  #end __init__

  def log(self, msg: str) -> None:
    """
    Log messages
    """
    log_msg = f'[info] {msg}'
    print(log_msg, file=self.files['msg'])
  #end log

  def debug(self, msg: str, description: str = "") -> None:
    """
    For debugging.
    Should have the same output file as the `log` method.
    """
    dbg_msg = f'[debug] {msg}'
    print(dbg_msg, file=self.files['msg'])
  #end debug

  def error(self, msg: str) -> None:
    """
    Print error messages.
    """
    err_msg = f'[error] {msg}'
    print(err_msg, file=self.files['err'])
  #end error

  def custom(self, tag: str):
    """
    For custom logging.
    """
    def custom_msg(msg: str) -> None:
      cus_msg = f'{tag} {msg}'
      print(cus_msg, file=self.files['msg'])
      #end custom_msg
    return custom_msg
  #end custom

  def blank_line(self, i: int=1) -> None:
    """
    Only for log.
    Should not be used in error logging and others.
    """
    print("", file=self.files['msg'])
  #end blank_line

#end class Logger

logger = Logger()






class Painter(object):
  def __init__(self, logger: Logger=logger):
    self.logger = logger

  def plot(self, xs: List[int], ys: List[int], style='.-') -> None:
    plt.figure()
    plt.plot(xs, ys, style)
    plt.grid()
  #end plot

  def draw_sample(self, batch):
    draw_sample(batch)

  def draw_seg_result(self, batch, pred, mask):
    show_single_figure_result(batch, pred, mask)

  def save_figure(self, path: str) -> None:
    try:
      plt.savefig(path)
      self.logger.log(f'saved figure {path}.')
    except IOError:
      self.logger.error(f'Trying to save figure {path} failed: {IOError}')
    #end save_figure
  #end save_figure

#end class Painter
painter = Painter()
