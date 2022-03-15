"""
Typing toolbox
"""
from collections import namedtuple
from typing_extensions import Annotated
from typing import (
  Any,
  List,
  Set,
  Dict,
  Tuple,
  Optional as Option,
  Union,
  Callable as Fn, 
  Iterator,
  Mapping, 
  MutableMapping, 
  Sequence,
  Match, 
  AnyStr, 
  IO,
  TypeVar,
  Generic,
  NewType,
)

def doc(s): pass

# Any    = 'Any'
Type   = 'Type'
# List   = namedtuple('List', 'T')
# Set    = namedtuple('Set', 'T')
# Dict   = namedtuple('Dict', 'Key Value')
# Tuple  = namedtuple('Tuple', 'First Second')
# Option = namedtuple('Option', 'T')
# Union  = namedtuple('Union', 'Left Right')
# Fn0    = namedtuple('Fn0', 'T')
# Fn1    = namedtuple('Fn1', 'A T')
# Fn2    = namedtuple('Fn2', 'A B T')
# Fn3    = namedtuple('Fn3', 'A B C T')
# Fn4    = namedtuple('Fn4', 'A B C D T')
# Fn5    = namedtuple('Fn5', 'A B C D E T')
# Fn6    = namedtuple('Fn6', 'A B C D E F T')
# Fn7    = namedtuple('Fn7', 'A B C D E F G T')

# def Function(*ts) -> Type:
#   func: Dict(int, 'HKT') = {
#     1: lambda ts: Fn0(ts[0]),
#     2: lambda ts: Fn1(ts[0], ts[1]),
#     3: lambda ts: Fn2(ts[0], ts[1], ts[2]),
#     4: lambda ts: Fn3(ts[0], ts[1], ts[2], ts[3]),
#     5: lambda ts: Fn4(ts[0], ts[1], ts[2], ts[3], ts[4]),
#     6: lambda ts: Fn5(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]),
#     7: lambda ts: Fn6(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5], ts[6]),
#     8: lambda ts: Fn7(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5], ts[6], ts[7]),
#   }
#   return func[len(ts)](ts)
#   #end Function

# Fn = Function
# Iterator = namedtuple('Iterator', 'T')
# Seq = namedtuple('Seq', 'T')
# IO = namedtuple('IO', 'T')
# Annotate = namedtuple('Annotate', 'str')

