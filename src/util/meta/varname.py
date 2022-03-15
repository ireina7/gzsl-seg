from util.syntax import *


def varname(var, dir):
  dir = locals()
  print(dir)
  return [ key 
    for key, val in dir.items() if id(val) == id(var)
  ]
end(varname)

