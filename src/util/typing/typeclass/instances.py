from util.syntax import *
from util.typing.basic import *
from util.typing.typevars import *
from util.typing.typeclass.typeclass import *





"""
Defaults
"""
class DefaultIterable(Default[Iterable]):
  def default(self) -> Iterable:
    return Iterable()
  end
end(DefaultIterable)

class DefaultShow(Default[Show]):
  def default(self) -> Show:
    return Show()
  end
end(DefaultShow)

class DefaultDebugging(Default[Debugging]):
  def default(self) -> Debugging:
    return Debugging()
  end
end(DefaultDebugging)

instance(Default[Iterable], DefaultIterable())
instance(Default[Show], DefaultShow())
instance(Default[Debugging], DefaultDebugging())



"""
Monoids
"""
class MonoidStr(Monoid[str]):
  def add(self, a, b): return a + b
  def zero(self): return ""
end(MonoidStr)

class MonoidInt(Monoid[int]):
  def add(self, a, b): return a + b
  def zero(self): return 0
end(MonoidInt)

class MonoidList(Monoid[List[T]]):
  def add(self, a: List[T], b: List[T]) -> List[T]:
    return a + b
  end(add)

  def zero(self) -> List[T]:
    return []
  end(zero)
end(MonoidList)


instance(Monoid[str], MonoidStr())
instance(Monoid[int], MonoidInt())
instance(Monoid[List[T]], MonoidList())





"""
Functors
"""
class FunctorList(Functor[List]):
  def map(self, f: Fn[[A], B], ma: List[A]) -> List[B]:
    return [f(a) for a in ma]
  end(map)
end(Functor)


instance(Functor[List], FunctorList())




"""
Applicatives
"""
class ApplicativeList(Applicative[List]):
  def map(self, f, ma):
    return summon(Functor[List]).map(f, ma)
  end(map)
  def pure(self, a: A) -> List[A]: return [a]
  def apply(self, ff: List[Fn[[A], B]], fa: List[A]) -> List[B]:
    pass
  end(apply)
end(ApplicativeList)


instance(Applicative[List], ApplicativeList())








"""
Iterables
"""
class IterableList(Iterable[List]):
  def __init__(self):
    return super().__init__()
  end(__init__)
end(IterableList)


instance(Iterable[List], IterableList())



"""
Show
"""
show = summon(Default[Show]).default()
instance(Show[str], show)
instance(Show[int], show)
instance(Show[float], show)
instance(Show[List[T]], show)
instance(Show[Dict[K, V]], show)



"""
Debugging
"""
debugging = summon(Default[Debugging]).default()
instance(Debugging[str],        debugging)
instance(Debugging[int],        debugging)
instance(Debugging[float],      debugging)
instance(Debugging[List[T]],    debugging)
instance(Debugging[Dict[K, V]], debugging)












# import util.typing.data as data
# """
# Tests
# """
# def test_monoids(xs: List[T],
#   T: data.Ref[type] = data.null,
#   monoid: data.Ref[Monoid[T]] = data.null
# ):
#   T = T or type(xs[0])
#   monoid = monoid or summon(Monoid[T])
#   acc = monoid.zero()
#   for x in xs:
#     acc = monoid.add(acc, x)
#   end
#   return acc
# end(test_monoids)

# Person = namedtuple('Person', 'name age id')

# class MonoidPerson(Monoid[Person]):
#   def zero(self) -> Person:
#     return Person('', 0, 0)
#   end
#   def add(self, a: Person, b: Person) -> Person:
#     return Person(a.name + b.name, a.age + b.age, a.id)
#   end
# end

# instance(Monoid, Person, MonoidPerson())


# def test_typeclass_pool() -> None:
#   print(gtcp.pool)
#   m = test_monoids(['1', '2', '3'], T=str)
#   n = test_monoids([1, 2, 3], T=int)
#   o = test_monoids([[1, 2, 3], [4, 5, 6]], T=List[int])
#   p = test_monoids([Person('Tom', 2, 0), Person('Ireina', 1, 1)], T=Person)
#   print(m)
#   print(n)
#   print(o)
#   print(p)

#   f = summon(Functor[List]).map(lambda x: x * 2, [1, 2, 3])
#   print(f) # [2, 4, 6]
  
# end(test_typeclass_pool)

