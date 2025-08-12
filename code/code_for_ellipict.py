#!/usr/bin/env python
from firedrake import *

mesh = UnitSquareMesh(8, 8)
V    = FunctionSpace(mesh, "CG", 1)

u  = TrialFunction(V)
v  = TestFunction(V)
f  = Constant(1.0)

a = dot(grad(u), grad(v))*dx
L = f*v*dx
bc = DirichletBC(V, 0.0, "on_boundary")

uh = Function(V)
solve(a == L, uh, bcs=bc)

print("‣ solution L2-norm =", norm(uh))
File("uh.pvd").write(uh)
