

all = [
  'iter',
  'error',
  'ref',
]

from util.typing.data.iter import Iter
from util.typing.data.error import *
from util.typing.data.ref import Ref, null



from util.typing.typeclass import *
"""
Tests
"""
def test_monoids(xs: List[T],
  T: type = None,
  monoid=None
):
  T = T or type(xs[0])
  (
    monoid,
  ) = (
    satisfy(monoid, Monoid[T]),
  )

  acc = monoid.zero()
  for x in xs:
    acc = monoid.add(acc, x)
  end
  return acc
end(test_monoids)

Person = namedtuple('Person', 'name age id')

class MonoidPerson(Monoid[Person]):
  def zero(self) -> Person:
    return Person('', 0, 0)
  end
  def add(self, a: Person, b: Person) -> Person:
    return Person(a.name + b.name, a.age + b.age, a.id)
  end
end

instance(Monoid[Person], MonoidPerson())


def test_typeclass_pool() -> None:
  gtcp.show()
  m = test_monoids(['1', '2', '3'], T=str)
  n = test_monoids([1, 2, 3], T=int)
  o = test_monoids([[1, 2, 3], [4, 5, 6]], T=List[int])
  p = test_monoids([Person('Tom', 2, 0), Person('Ireina', 1, 1)], T=Person)
  print(m)
  print(n)
  print(o)
  print(p)

  # gtcp.show()
  f = summon(Functor[List]).map(lambda x: x * 2, [1, 2, 3])
  print(f) # [2, 4, 6]
  
end(test_typeclass_pool)



