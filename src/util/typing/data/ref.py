from util.syntax import *
from util.typing.basic import *
# from util.typing.typeclass.core import instance
from util.typing.typevars import *
from util.typing.typeclass import Reference, Extractor, Default, instance, summon


class Ref(Generic[A]):
  def __init__(self, a: A):
    self.value = a
  end(__init__)
end(Ref)


class ReferenceRef(Reference[Ref]):
  def modify(self, f: Fn[[A], B], ref: Ref[A]) -> None:
    ref.value = f(ref.value)
  end(modify)
end(ReferenceRef)

# This is evil
class ExtractorRef(Extractor[Ref]):
  def extract(self, ref: Ref[A]) -> A:
    return ref.value
  end(extract)
end(ExtractorRef)

class DefaultRef(Default[Ref[A]]):
  def default(self) -> Ref[A]:
    return Ref(None)
  end(default)
end(DefaultRef)


instance(Reference[Ref], ReferenceRef())
instance(Extractor[Ref], ExtractorRef())
instance(Default[Ref], DefaultRef())

null = summon(Default[Ref[A]]).default()
