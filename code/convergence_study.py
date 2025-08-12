from mixed_elasticity import *

import re
from firedrake.petsc import PETSc
print = PETSc.Sys.Print

opts = PETSc.Options()
# Read runtime options, these are passed by running the script as
# python [script.py] -[first_option_name] [first_option_value] -[second_option_name] [second_option_value]
#
# Option values may be a comma-separated list.

# Always use the Constant type for physical parameters so that the generated
# code does not change when the parameter changes its value

# Poisson ratio
nu = Constant(0, name="nu")

# Young's modulus
E = Constant(1, name="E")

# Read the number of elements along an edge of the base mesh
nx = opts.getInt("nx", 4)

# Read the number of alpha
#alpha = opts.getInt("alpha", 2)

# Read the number of levels of mesh refinement
refine = opts.getInt("refine", 1)

# Read the degree range
degrees = list(map(int, opts.getIntArray("degree", (3,))))

# Read the Poisson ratios from option -nu, here we also set the default value to 0.3
nu_range = opts.getRealArray("nu", (0.3,))
param_ranges = {
    "nu": nu_range,
}

# Augmented Lagrangian coefficient
fgamma = opts.getReal("gamma", 1E3)
gamma = Constant(fgamma, name="gamma")

#base = UnitSquareMesh(nx, nx, diagonal="crossed")
base = UnitSquareMesh(nx, nx) # three lines mesh
meshes = MeshHierarchy(base, refine)

# Read the problem type ['neohookean', 'constant', 'inclusion', 'dg5']
problem = opts.getString("problem", "constant")
nonlinear = opts.getBool("nonlinear", False)
exact = opts.getString("exact", "sin") 


class InclusionSolver(LinearElasticitySolver):

    def material_parameters(self, mesh):
        mu, rlam = super().material_parameters(mesh)
        x, y = SpatialCoordinate(mesh)
        hat = conditional(ufl.And(abs(x-1/2) < 1/4, abs(y-1/2) < 1/4), 1, 0)
        rlam = rlam * (1-hat)
        mu = E/3 * hat +(1-hat)* mu
        return mu, rlam

params = (nu, E, gamma)
if problem == "neohookean":
    solver = NeoHookeanSolver(*params, nonlinear=nonlinear, exact=exact)
elif problem == "ogden":
    solver = OgdenSolver(*params, nonlinear=nonlinear,exact=exact)
elif problem == "constant" :
    solver = LinearElasticitySolver(*params,exact=exact)
elif problem == "inclusion":
    solver = InclusionSolver(*params,exact=exact)
else:
    raise ValueError("Unknown problem type")

solver_parameters = {
    "augmented_lagrangian": True,
    "mat_type": "matfree",
    "pmat_type": "nest",
    "ksp_type": "minres",
    "ksp_rtol": 1E-12,
    "ksp_atol": 1E-14,
    #"ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_pc_type": "lu",
    "fieldsplit_pc_factor_mat_solver_type": "mumps",
}
solver_parameters = None
print(f"Problem type: {problem}") 
print(f"Nonlinear: {nonlinear}") 
print(f"Exact: {exact}")
if problem =="neohookean" or "odgen":
    if nonlinear:
        error_types = "errorDisplacement,errorPressure,errorStress,errorIncompressibility"
    else:
        error_types = "errorDisplacement,errorPressure"
else:
    error_types = "errorStandard,errorStress"

# Nested loop on degrees and refinement levels
for degree in degrees:
    filename = f"{problem}_{nonlinear}_{exact}_degree{degree}.csv"
    emptyfile = True

    print(f"Solving for nu = {nu_range} with degree {degree} with {len(meshes)-1} refinements")
    for level, mesh in enumerate(meshes):
        print(f"Solving {degree=} {level=}")
        if level == 1 and nonlinear:
            # Switch off continuation if we are not the coarse grid
            param_ranges = {pname: (pvals[-1],) for pname, pvals in param_ranges.items()}

        errors_list, dofs = solver.solve_mms(mesh, degree,
                                             param_ranges=param_ranges,
                                             solver_parameters=solver_parameters)

        # Save errors to file
        if emptyfile:
            emptyfile = False
            if PETSc.COMM_WORLD.rank == 0:
                with open(filename, "w") as f:
                    param_names = ",".join(list(solver.params))
                    f.write(f"degree,dofs,level,{param_names},{error_types}\n")
        for (pvals, errors) in errors_list:
            if PETSc.COMM_WORLD.rank == 0:
                with open(filename, "a") as f:
                    csv_errors = ",".join(map(str, [*pvals, *errors]))
                    f.write(f'{degree},{dofs},{level},{csv_errors}\n')
        print()
    print()
