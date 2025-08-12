from firedrake import *
from ufl.domain import as_domain
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
from firedrake.mg.utils import get_level

import numpy

print = PETSc.Sys.Print

opts = PETSc.Options()
# Read runtime options, these are passed by running the script as
# python [script.py] -[first_option_name] [first_option_value] -[second_option_name] [second_option_value]
#
# Option values may be a comma-separated list.

# Always use the Constant type for physical parameters so that the generated
# code does not change when the parameter changes its value

# Poisson ratio
nu = Constant(0)

# Young's modulus
E = Constant(1)

# Read the number of elements along an edge of the base mesh
nx = opts.getInt("nx", 4)

# Read the number of levels of mesh refinement
refine = opts.getInt("refine", 1)

# Read the degree range
degrees = list(map(int, opts.getIntArray("degree", (3,))))

# Read the Poisson ratios from option -nu, here we also set the default value to 0.3
nu_range = opts.getRealArray("nu", (0.3,))

base = UnitSquareMesh(nx, nx, diagonal="crossed")
meshes = MeshHierarchy(base, refine)

fgamma = opts.getReal("gamma", 1E3)
gamma = Constant(fgamma)

# Read the problem type
problem = opts.getString("problem", "constant")


def eps(u):
    """Returns the symmetric part of the gradient.
    """
    return sym(grad(u))


def as_mixed(functions):
    """Returns a concatenation of the given list of functions.
    """
    components = []
    for f in functions:
        components.extend(f[idx] for idx in numpy.ndindex(f.ufl_shape))
    return as_vector(components)


class MixedLinearElasticitySolver():

    def __init__(self, *params):
        self.params = params
        self.solvers = {}

    def make_function_space(self, mesh, degree):
        """Returns the FunctionSpace.
        """
        V = VectorFunctionSpace(mesh, "CG", degree)
        Q = FunctionSpace(mesh, "DG", degree-1)
        Z = V * Q
        return Z

    def material_parameters(self, mesh):
        nu, E = self.params
        x = SpatialCoordinate(mesh)
        # Lame parameters
        mu = E / (2 * (1 + nu))
        #lam = E * nu / ((1 + nu) * (1 - 2*nu))
        lam_inverse =  ((1 + nu) * (1 - 2*nu))/(E * nu)
        return mu, lam_inverse

    def stress(self, u, p):
        mesh = as_domain(u)
        d = mesh.geometric_dimension()
        mu, lam = self.material_parameters(mesh)
        sigma = 2 * mu * eps(u) + p * Identity(d)
        return sigma

    def make_form(self, Z):
        """Returns the bilinear form.
        """
        u, p = TestFunctions(Z)
        v, q = TrialFunctions(Z)
        mu, rlam = self.material_parameters(Z.mesh())
        # TODO modify the bilinear form to solve the mixed elasticity problem
        a = inner(2*mu*eps(u), eps(v)) * dx + inner(p, div(v)) * dx + inner(q, div(u)) * dx - inner(p * rlam, q) * dx
        return a

    def make_bcs(self, Z, ubc=0):
        """Returns the boundary conditions.
        """
        # subs = [(1, 2, 3)]  # impose the BC on part of the boundary
        subs = ["on_boundary"]  # impose the BC on the entire boundary
        # Use Z.sub(0) to impose BCs only on the displacement
        bcs = [DirichletBC(Z.sub(0), ubc, sub) for sub in subs]
        return bcs

    def exact_solution(self, mesh):
        """Returns the symbolic representation of the exact solution.
        """
        mu, rlam = self.material_parameters(mesh)
        # Symbolic mesh coordinates
        x = SpatialCoordinate(mesh)
        # Return a vector valued expression
        u_incomp = curl((sin(x[0]*pi)**2) * (sin(x[1]*pi)**2))
        u_comp = x * sin(x[0]*pi) * sin(x[1]*pi)
        uex = conditional(abs(rlam) < 1E-12, u_incomp, u_comp)
        # pressure on the incompressible region
        p0 = cos(x[0]*2*pi) * cos(x[1]*2*pi)
        pex = conditional(abs(rlam) < 1E-12, p0, 1/rlam * div(u_comp))
        return uex, pex

    def error_norms(self, uh, uex, ph, pex, a=None):
        """Computes the error norms
        """
        #if a is None:
        #    a = lambda v, u: inner(self.stress(u), eps(v))*dx

        #err_energy = (assemble(a(uh-uex, uh-uex)) / assemble(a(uex, uex))) ** 0.5
        hypot = lambda x, y: (x**2 + y**2)**0.5

        err_standard = hypot(norm(uh-uex, 'H1'), norm(ph-pex)) / hypot(norm(uex, 'H1'), norm(pex))
        err_stress = norm(self.stress(uh, ph)-self.stress(uex, pex), 'L2') / norm(self.stress(uex, pex), 'L2')
        return err_standard, err_stress

    def solve_mms(self, mesh, degree, solver_parameters=None):
        """Uses the method of manufactured solutions to setup and solve a
           variational problem and returns the error norms.
        """
        _, level = get_level(mesh)
        # Construct an exact solution
        uex, pex = self.exact_solution(mesh)
        zex = as_mixed([uex, pex])

        # Construct the discrete function space
        Z = self.make_function_space(mesh, degree)
        try:
            solver = self.solvers[Z]
        except KeyError:
            # Setup the bilinear form
            a = self.make_form(Z)
            test, trial = a.arguments()

            # Construct the right-hand side and boundary conditions such that uex is the
            # exact solution
            L = a(test, zex)
            bcs = self.make_bcs(Z, ubc=uex)

            u, p = split(trial)
            v, q = split(test)
            mu, rlam = self.material_parameters(mesh)

            rrho = rlam + 1/gamma
            rho = 1/rrho
            aP = (inner(2*mu*eps(u), eps(v)) + inner(div(u)*rho, div(v)))*dx + inner(p*rrho, q)*dx

            # Create a discrete Function uh, initialized to 0
            zh = Function(Z)
            problem = LinearVariationalProblem(a, L, zh, bcs=bcs, aP=aP)
            solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix="")
            self.solvers[Z] = solver

        # Solve the variational problem, and store the solution in uh
        solver.solve()

        zh = solver._problem.u
        uh, ph = zh.subfunctions

        # Get the number of degrees of freedom
        dofs = Z.dim()

        # correct the pressure mean value
        vol = assemble(1*dx(domain=mesh))
        p0 = (1/vol)*assemble((pex-ph)*dx)
        ph.assign(ph + p0)

        #import matplotlib.pyplot as plt
        #trisurf(uh)
        #plt.show()

        VTKFile(f"output/solution_{level}.pvd").write(uh, ph)

        # Compute the error norms
        err_standard, err_stress = self.error_norms(uh, uex, ph, pex)
        return err_standard, err_stress, dofs 
        #return self.error_norms(uh, uex, ph, pex)

class InclusionSolver(MixedLinearElasticitySolver):
    def material_parameters(self, mesh):

        mu,rlam = super().material_parameters(mesh)
        x, y = SpatialCoordinate(mesh)
        hat = conditional(ufl.And(abs(x-1/2) < 1/4, abs(y-1/2) < 1/4), 1, 0)
        rlam = rlam * (1-hat)
        return mu, rlam

if problem == "constant" :  
    solver = MixedLinearElasticitySolver(nu, E)
elif problem == "inclusion":
    solver = InclusionSolver(nu,E)

solver_parameters = {
    "mat_type": "matfree",
    "pmat_type": "nest",
    "ksp_type": "minres",
    "ksp_rtol": 1E-14,
    "ksp_atol": 0,
    #"ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_pc_type": "lu",
    "fieldsplit_pc_factor_mat_solver_type": "mumps",
}

# Nested loop on degrees and refinement levels
for degree in degrees:
    print(f"Solving for nu = {nu_range} with degree {degree} with {len(meshes)-1} refinements")

    filename = f"{problem}_degree{degree}.csv"
    if PETSc.COMM_WORLD.rank == 0:
        with open(filename, "w") as f:
            f.write("degree,level,nu,dofs,errorStandard,errorStress\n")

    for level, mesh in enumerate(meshes):
        for nu_val in nu_range:
            nu.assign(nu_val)
            errors = solver.solve_mms(mesh, degree, solver_parameters=solver_parameters)
            err_standard, err_stress, dofs = errors
            nu_errors = (nu_val, dofs, *errors[:2])
            #nu_errors = (nu_val, *errors)
            print(f"{degree=}, {level=}, nu={float(nu)}, {dofs=}, {err_standard=}, {err_stress=}")

            # Save errors to file
            if PETSc.COMM_WORLD.rank == 0:
                with open(filename, "a") as f:
                    csv_errors = ",".join(list(map(str, nu_errors)))
                    f.write(f'{degree},{level},{csv_errors}\n')
        print()
    print()
