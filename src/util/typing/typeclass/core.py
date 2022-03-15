from util.syntax import *
from util.typing.basic import *
from util.typing.typevars import *
from util.meta import varname



TypeClass = Generic

class TypeclassSystem(object):
  """
  Typeclass system interface.
  """
  def query(self, tc, tag) -> doc('Instance of tc with tag'):
    """Query the corresponding instance
    """
    pass
  end(query)

  def add_typeclass(self, tc) -> None:
    """Add one typeclass
    """
    pass
  end(add_typeclass)

  def add_instance(self, tc, tag, instance) -> None:
    """Add one instance
    """
    pass
  end(add_instance)
end(TypeclassSystem)




class TypeclassPool(TypeclassSystem):
  """
  A naive implementation of typeclass system.
  """
  def __init__(self):
    super(TypeclassPool, self).__init__()
    self.pool = {}
  end(__init__)

  def add_typeclass(self, tc) -> None:
    if tc in self.pool:
      return
    else:
      self.pool[tc] = {}
    end
  end(add_typeclass)

  def add_instance(self, tc, tag, instance) -> None:
    if not tc in self.pool:
      self.add_typeclass(tc)
    end
    self.pool[tc][tag] = instance
  end(add_instance)

  def query(self, tc, query) -> doc('Option[Instance of tc with tag]'):
    """Query an instance by Typeclass[query]
    @param tc: the type class name
    @param query: the type registered within the typeclass
    @return: The corresponding instance | None
    @note: we also handled the genertic types.
    """
    if not tc in self.pool:
      return None
    end
    for key in self.pool[tc]: # this is bad, evil, stupid!
      if self.match(key, query):
        return self.pool[tc][key]
      end
    end
    return None
  end(query)

  def query_type(self, tc, x) -> doc('Instance of tc of x'):
    """Query directly by inferring the type of value x(not safe due to generic type)
    @param tc: the typeclass
    @param x: the value
    @return: the corresponding instance
    """
    return self.query(tc, type(x))
  end(query_type)

  def is_generic(self, t):
    """Test if type is a generic type (i.e., has type parameters, like Monoid[int])
    @param t: the tested type
    @return: True or False
    """
    try:
      t.__args__
    except AttributeError:
      return False
    end
    return True
  end(is_generic)

  def match_type(self, tag, query_t):
    """Check if two types match each other.
    @param tag: the type in typeclass pool
    @param query_t: the queried type
    @return: True | False
    """
    if tag == query_t:
      return True
    end
    if self.is_generic(query_t) and (not self.is_generic(tag)):
      return query_t.__origin__ == tag
    end
    if not (self.is_generic(tag) and self.is_generic(query_t)):
      # if both are not generic and they are not equal, then return false.
      return False
    end
    # now both are generic types
    if tag.__origin__ != query_t.__origin__:
      return False
    end
    # now we check type parameters
    xs = tag.__args__
    ys = query_t.__args__
    if not len(xs) == len(ys):
      return False
    end
    for a, b in zip(xs, ys):
      is_ok = self.match_type(a, b) or isinstance(a, TypeVar) or isinstance(b, TypeVar)
      if not is_ok: return False
    end
    return True
  end(match_type)

  def match(self, tag, query_t):
    """Match two sequences of types (BOTH ARE SEQUENCES!!)
    @param tag: the types in the pool
    @param query_t: the query type sequence
    @return: True | False
    """
    if not len(tag) == len(query_t):
      return False
    end
    for a, b in zip(tag, query_t):
      if not self.match_type(a, b): return False
    end
    return True
  end(match)

  def show(self) -> None:
    """Show the typeclass pool to debug.
    @return: None
    """
    for typeclass in self.pool:
      print(f'Typeclass: {typeclass}:')
      for t in self.pool[typeclass]:
        instance = self.pool[typeclass][t]
        print(f'\t{t}: {instance}')
      end
    end
  end(show)

end(TypeclassPool)


global_typeclass_pool = TypeclassPool()
gtcp = global_typeclass_pool # for convenience




def summon_type(tc):
  """Summon directly by pass type parameters.
  @example: summon_type(Monoid)(int)
  """
  def search(t):
    if not isinstance(t, tuple):
      t = (t,)
    result = gtcp.query(tc, t)
    if result == None:
      raise Exception(f'summon instance failed: searching instance of {tc}[{t}]')
    end
    return result
  end(search)
  return search
end(summon_type)

def summon_value(tc):
  """Summon by value
  @example: summon_value(Monoid)(7)
  """
  return lambda x: summon_type(tc)(type(x))
end(summon_value)

def summon(tc):
  """Summon by complete type annotation.
  @example: summon(Monoid[int])
  """
  # print(tc.__origin__, tc.__args__)
  typeclass = tc.__origin__
  tag = tc.__args__
  result = gtcp.query(typeclass, tag)
  import util.typing.data as data
  if result == None:
    raise data.SummonNotFound(
      typeclass, tag, 
      f'summon instance failed: searching instance of {tc}'
    )
  end
  return result
end(summon)


def typeclass(tc):
  """Add typeclass
  @example: typeclass(Monoid)
  """
  gtcp.add_typeclass(tc)
end(typeclass)

# def instance(tc, tag, ins):
#   if not isinstance(tag, tuple):
#     tag = (tag,)
#   end
#   gtcp.add_instance(tc, tag, ins)
# end(instance)

def instance(tc, ins):
  """Add instance
  @example: instance(Monoid[int], monoid_int_instance)
  """
  if not gtcp.is_generic(tc):
    raise Exception(f'Error while registering instance of {tc} with {ins}')
  tag = tc.__args__
  tc = tc.__origin__
  gtcp.add_instance(tc, tag, ins)
end(instance)

def satisfy(x, tc: type):
  return x or summon(tc)
end(satisfy)



