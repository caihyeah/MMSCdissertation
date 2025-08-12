# Ogden Hyperelasticity FEM in Firedrake
from firedrake import *
import ufl

# Mesh and function spaces
mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, "CG", 2)  # displacement space
Q = FunctionSpace(mesh, "CG", 1)        # pressure space
W = V * Q

# Trial and test functions
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# Deformation gradient and invariants
I = Identity(mesh.geometric_dimension())
F = I + grad(u)
C = F.T * F
J = det(F)

# Ogden model parameters (N=1)
mu = Constant(1.0)
alpha = Constant(2.0)

# Compute principal stretches (approximate: isotropic approximation)
I1 = tr(C)
I1_sqrt = sqrt(I1)
W_iso = (mu / alpha) * (I1_sqrt**alpha - 3.0)  # approximate Ogden form

# Total energy
W_total = W_iso  # Incompressibility handled separately

# First Piola–Kirchhoff stress tensor with incompressibility
P = ufl.diff(W_total, F) + p * inv(F).T

# Weak form
residual_eq = inner(P, grad(v)) * dx - dot(Constant((0, 0)), v) * dx
residual_incomp = q * (J - 1.0) * dx
F_form = residual_eq + residual_incomp

# Boundary conditions
bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary")

# Nonlinear problem and solver
solve(F_form == 0, w, bcs=[bc], solver_parameters={
    "nonlinear_solver": "snes",
    "snes_monitor": None,
    "snes_type": "newtonls",
    "mat_type": "aij",
    "pc_type": "lu"
})

# Post-process solution
(u_sol, p_sol) = w.split()
File("displacement.pvd").write(u_sol)
File("pressure.pvd").write(p_sol)
