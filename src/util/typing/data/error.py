import typing
from util.syntax import *
from util.typing.basic import *
from util.typing.typevars import *


class Error(Exception):
  pass
end(Error)


class SummonNotFound(Error):
  def __init__(self, typeclass, tag, msg: str):
    self.typeclass = typeclass
    self.tag = tag
    self.message = msg
    super().__init__(msg)
  end(__init__)
end(SummonNotFound)


