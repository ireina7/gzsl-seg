from util.syntax import *
from util.typing.basic import *
from util.typing.torch import *
from util.typing.typeclass.typeclass import *
from util.typing.typevars import *
from util import *


# class Dataset(Iterator):
#   def __init__(self):
#     pass
# #end DatasetIterator

class DataBatch(TypeClass[A]):
  def info(self, batch: A): pass
  def data(self, batch: A): pass
  def label(self, batch: A): pass
end(DataBatch)

class SwitchDevice(TypeClass[A]):
  def switch(self, device, data): pass
end(SwitchDevice)

class Optimizable(TypeClass[A]):
  def zero_grad(self, optimizer: A) -> None:
    optimizer.zero_grad()
  end(zero_grad)
  def step(self, optimizer: A) -> None:
    optimizer.step()
  end(step)
end(Optimizable)

instance(Default[Optimizable], Default(Optimizable))

Frame = namedtuple('Frame', 'epoch i model batch pred loss')

class Device:
  def __init__(self, device):
    self.raw_device = device
  end(__init__)

  def name() -> str: pass
end(Device)

class DeepLearning(TypeClass[A]):
  
  def parameters(self, model: A):
    return model.parameters()
  end
  def train(self, model: A):
    model.train()
  end
  def eval(self, model: A):
    model.eval()
  end
  def forward(self, model: A, batch):
    model.forward()
  end
  def backward(self, model: A):
    model.backward()
  end
end(DeepLearning)

instance(Default[DeepLearning], Default(DeepLearning))





class Experiment(object):
  """
  Define what an experiment is.
  @date 2022.01
  """
  def __init__(self, 
    name: str='Anonymous experiment',
    logger: Logger=logger,
    args={}
  ):
    self.name = name
    self.args = args
    self.logger = logger
  end(__init__)


  Dataset   = TypeVar('Dataset')
  Model     = TypeVar('Model')
  Optimizer = TypeVar('Optimizer')
  Batch     = TypeVar('Batch')
  def train(self, 
    model: Model, 
    dataset: Dataset,
    loss_fn: Fn, 
    optimizer: Optimizer,
    epochs: int = 10,
    Model: type = None,
    Dataset: type = None,
    Optimizer: type = None,
    Batch: type = None,
    dataset_Iterable: Iterable = None,
    model_NeuralModel: DeepLearning = None,
    data_SwitchDevice: SwitchDevice = None,
    model_SwitchDevice: SwitchDevice = None,
    optimizer_Optimizable: Optimizable = None,
    batch_DataBatch: DataBatch = None,
  ) -> doc('Trained Model'):
    """
    Training the model.
    """
    Model = Model or type(model)
    Dataset = Dataset or type(dataset)
    Optimizer = Optimizer or type(optimizer)
    ( # Typing and reinitialization
      dataset_dsl, 
      model_dsl,
      data_device_dsl,
      model_device_dsl,
      optimizer_dsl,
      batch_dsl,
    ) = (
      satisfy(dataset_Iterable, Iterable[Dataset]),
      satisfy(model_NeuralModel, DeepLearning[Model]),
      satisfy(data_SwitchDevice, SwitchDevice[Tensor]),
      satisfy(model_SwitchDevice, SwitchDevice[Model]),
      satisfy(optimizer_Optimizable, Optimizable[Optimizer]),
      satisfy(batch_DataBatch, DataBatch[Batch]),
    )

    # Domain specific languages
    args         = self.args
    device       = args.device
    zero_grad    = optimizer_dsl.zero_grad
    iterator     = dataset_dsl.iter
    data         = batch_dsl.data
    label        = batch_dsl.label
    forward      = model_dsl.forward
    backward     = model_dsl.backward
    step         = optimizer_dsl.step
    switch_model = model_device_dsl.switch
    switch_data  = data_device_dsl.switch
    prepare      = model_dsl.train
    # input_size = args.input_size

    # Real logic
    prepare(model)
    switch_model(device, model)
    zero_grad(optimizer)

    self.peek_before_train_start(args)
    for epoch in range(epochs + 1):
      self.peek_per_epoch(epoch)
      iter = iterator(dataset)
      prepare(model)

      for i in range(len(dataset)):
        batch = iter.next()
        images, masks = data(batch), label(batch)
        images = switch_data(device, images)
        masks  = switch_data(device, masks.long())
        pred = forward(model, images)
        loss = loss_fn(pred, masks)
        backward(model)
        step(optimizer)

        self.peek(model, pred, batch)
      end
      if args.eval_if(epoch):
        self.eval(model)
      end
    end
    self.peek_after_train()
  end(train)

  def peek(self, frame: Frame) -> doc('Peek results'):
    pass
  end(peek)

  def peek_before_train_start(self, args):
    print_config(args)
    self.logger.blank_line(2)
    self.logger.log('Training start ...')
  end(peek_before_train_start)

  def peek_per_epoch(self, frame: Frame):
    self.logger.log(f"[Epoch: {frame.epoch}]")
  end(peek_per_epoch)

  def peek_after_train(self):
    self.logger.log('[Training end.]')
  end

  def eval(self, model) -> doc('Evaluation results'):
    pass
  end(eval)
end(Experiment)
