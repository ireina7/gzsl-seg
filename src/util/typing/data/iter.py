from util.syntax import *
from util.typing.basic import *
from util.typing.typevars import *


"""
Mutable object to interact with python.
"""
class Iter(Generic[A]):
  def __init__(self, iterator: Iterator[A]):
    self.iterator = iterator
  end(__init__)
  def next(self) -> Option[A]:
    x = None
    try:
      x = next(self.iterator)
    except StopIteration:
      x = None
    return x
  end(next)
  # Optional methods
  def __next__(self):
    return self.next()
  def peek(self) -> Option[A]: pass
end(Iter)



