from util.syntax import *
from util.typing.basic import *
from util.typing.typevars import *
from util.typing.typeclass.core import *


class TaggedValue:
  def tag(): pass
end(TaggedValue)

class Decorator(Generic[A]):
  def __call__(self, a: A): pass
end(Decorator)


class Reference(Generic[F]):
  """The effect of modifing value
  """
  # modify: (A -> B) -> F[A] -> None
  def modify(self, f: Fn[[A], B], ref): pass
end(Reference)




class Extractor(TypeClass[F]):
  def extract(self, container) -> A: pass
end(Extractor)

class Comonad(Extractor[F]):
  def coflat_map(self): pass
end


class Monoid(TypeClass[A]):
  def add(self, a: A, b: A) -> A: pass
  def zero(self) -> A: pass
end(Monoid)


class Functor(TypeClass[F]):
  # map: (A -> B) -> F[A] -> F[B]
  def map(self, f, ma): pass
end(Functor)

class Applicative(Functor[F]):
  # pure: A -> F[A]
  def pure(self, a: A): pass
  # apply: (F[A -> B]) -> F[A] -> F[B]
  def apply(self, ff, fa): pass
end(Applicative)

class Monad(Applicative[F]):
  # flat_map: F[A] -> (A -> F[B]) -> F[B]
  def flat_map(self, ma, f): pass
end(Monad)



"""
The typeclass to provide iters.
"""
class Iterable(Generic[F]):
  # iter: F[A] -> Iter[A]
  def iter(self, container):
    """
    Default implementation.
    """
    import util.typing.data as data
    raw_iter = container.__iter__()
    return data.Iter[A](raw_iter)
  end(iter)
end(Iterable)


"""
Show
"""
class Show(Generic[A]):
  def show(self, x: A) -> str:
    return x.__str__()
  end(show)
end(Show)


"""
Debugging
"""
class Debugging(Generic[A]):
  def debug(self, x: A) -> str:
    return x.__repr__()
  end(debug)
end(Debugging)


"""
Default
"""
class Default(Generic[A]):

  def __init__(self, constructor=None):
    self.constructor = constructor
  end(__init__)

  def default(self) -> A:
    if self.constructor != None:
      return self.constructor()
    else:
      return None
    end
  end(default)
end(Default)


"""
Eq
"""
class Equal(Generic[A]):

  def equals(self, x: A, y: A) -> bool:
    return x == y
  end(equals)

  def not_equals(self, x: A, y: A) -> bool:
    return not self.equals(x, y)
  end(not_equals)
end(Equal)


"""
Compare
"""
import enum
class ComparingResult(enum.Enum):
  Less = -1
  Equal = 0
  Greater = 1
end(ComparingResult)

class Comparing(Generic[A]):

  def compare(self, x: A, y: A) -> ComparingResult:
    pass
  end(compare)
end(Comparing)











"""
Register all typeclasses.
"""
typeclass(Monoid)
typeclass(Functor)
typeclass(Applicative)
typeclass(Monad)
typeclass(Iterable)



