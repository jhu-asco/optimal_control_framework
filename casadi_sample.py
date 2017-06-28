#!/usr/bin/env python

from casadi import *

x = MX.sym('x')
print "Jacobian of sinx: ", jacobian(sin(x), x)
