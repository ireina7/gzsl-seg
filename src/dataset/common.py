from util.typing.typeclass import *
from util.typing.torch import *
from util.experiment import *
from util.typing.data import Iter
from torch import Tensor




class SwitchDeviceTensor(SwitchDevice[Tensor]):
  def switch(self, device, data: Tensor):
    return data.to(device)
  end(switch)
end(SwitchDeviceTensor)





class Batch:
  def __init__(self, data, label, info=None):
    self.data = data
    self.label = label
    self.info = info
  end(__init__)
end(Batch)

class DataBatchBatch(DataBatch[Batch]):
  def info(self, batch: Batch):
    return batch.info
  end
  def data(self, batch: Batch):
    return batch.data
  end
  def label(self, batch: Batch):
    return batch.label
  end
end(DataBatchBatch)



class DataLoaderIter(Iter[Batch]):
  def __init__(self, raw_iter):
    self.raw_iter = raw_iter
  end
  def next(self):
    x = self.raw_iter.__next__()
    return Batch(
      x['image'], 
      x['label'], 
      info={'name': x['name'], 'size': x['size']}
    )
  end(next)
end(DataLoaderIter)

# Let's provided that the DataLoader class is actually a Higher-Kinded type...
class IterableDataLoader(Iterable[DataLoader]):
  # iter: DataLoader[Batch] -> Iter[Batch]
  def iter(self, dataloader):
    raw_iter = dataloader.__iter__()
    return DataLoaderIter(raw_iter)
  end(iter)
end(IterableDataLoader)


instance(DataBatch[Batch], DataBatchBatch())
instance(Iterable[DataLoader], IterableDataLoader())
instance(SwitchDevice[Tensor], SwitchDeviceTensor())



