from firedrake import *
from firedrake.output import VTKFile
from firedrake.mg.utils import get_level
from ufl.domain import as_domain
from ufl import replace, conditional, And
import numpy


class FunctionHierarchy():
    """Class to transfer Functions across levels of a mesh hierarchy.
    """
    def __init__(self, u):
        mesh = as_domain(u)
        mh, l = get_level(mesh)
        self.u = {l: u,}

    def get_function(self, mesh):
        mh, lf = get_level(mesh)
        lc = min(self.u)
        uf = self.u[lc]
        for l in range(lc+1, lf+1):
            uc = uf
            try:
                uf = self.u[l]
            except KeyError:
                Vc = uc.function_space()
                Vf = Vc.reconstruct(mh[l])
                uf = Function(Vf)
                self.u[l] = uf
            prolong(uc, uf)
        return uf


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


class HyperElasticitySolver():
    """Class to implement nonlinear or linearized hyperelasticity
    solvers using the method of manufactured solutions
    """
    def __init__(self, *params, nonlinear=False):
        self.params = params
        self.solvers = {}
        self.nonlinear = nonlinear

    def material_parameters(self, mesh):
        # Lame parameters
        nu, E = self.params[:2]
        mu = E / (2 * (1 + nu))
        rlam = ((1 + nu) * (1 - 2*nu)) / (E * nu)
        return mu, rlam

    def energy(self, z):
        raise NotImplementedError("Please implement the energy")

    def stress(self, z):
        raise NotImplementedError("Please implement the stress")

    def exact_solution(self, mesh):
        """Returns the symbolic representation of the exact solution.
        """
        mu, rlam = self.material_parameters(mesh)
        # Symbolic mesh coordinates
        x = SpatialCoordinate(mesh)
        # incompressible displacement
        A = Constant(1E-2)
        uex = curl(A * sin(x[0]*pi)**2 * sin(x[1]*pi)**2)
        # pressure on the incompressible region
        p0 = cos(x[0]*2*pi) * cos(x[1]*2*pi)
        pex = conditional(abs(rlam) < 1E-12, p0, 1/rlam * div(uex))
        return uex, pex

    def make_function_space(self, mesh, degree):
        """Returns the FunctionSpace.
        """
        V = VectorFunctionSpace(mesh, "CG", degree)
        Q = FunctionSpace(mesh, "DG", degree-1)
        Z = V * Q
        return Z

    def make_bcs(self, Z, ubc=0):
        """Returns the boundary conditions.
        """
        subs = ["on_boundary"]  # impose the BC on the entire boundary
        bcs = [DirichletBC(Z.sub(0), ubc, sub) for sub in subs]
        return bcs

    def make_nonlinear_residual(self, z, quadrature_degree=None):
        """Returns the nonlinear residual
        """
        U = self.energy(z, quadrature_degree=quadrature_degree)
        F = derivative(U, z)
        return F

    def make_bilinear_form(self, z, quadrature_degree=None):
        """Returns the Jacobian bilinear form
        """
        F = self.make_nonlinear_residual(z, quadrature_degree=None)
        J = derivative(F, z)
        return J

    def make_preconditioner(self, test, trial):
        """Returns the preconditioner bilinear form
        """
        mesh = as_domain(test)
        u, p = split(trial)
        v, q = split(test)
        mu, rlam = self.material_parameters(mesh)

        gamma = self.params[2]
        rrho = rlam + 1/gamma
        rho = 1/rrho
        aP = (((2*mu)*inner(eps(u), eps(v)) + rho*inner(div(u), div(v)))*dx
              + rrho*inner(p, q)*dx)
        return aP

    def make_linearized_solver(self, Z, zex, solver_parameters=None, quadrature_degree=None):
        """Returns a solver for the linearized hyperelasticity problem
        """
        uex, pex = zex
        bcs = self.make_bcs(Z, ubc=uex)
        # Create a discrete Function uh, initialized to 0
        z = Function(Z)
        a = self.make_bilinear_form(z, quadrature_degree=quadrature_degree)
        test, trial = a.arguments()
        L = a(test, as_mixed(zex))

        aP = self.make_preconditioner(test, trial)
        problem = LinearVariationalProblem(a, L, z, bcs=bcs, aP=aP)
        solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix="")
        return solver

    def make_nonlinear_solver(self, Z, zex, solver_parameters=None, quadrature_degree=None):
        """Returns a solver for the nonlinear hyperelasticity problem
        """
        uex, pex = zex
        bcs = self.make_bcs(Z, ubc=uex)
        # Create a discrete Function uh, initialized to 0
        z = Function(Z)
        F = self.make_nonlinear_residual(z, quadrature_degree=quadrature_degree)
        L = replace(F, {z: as_mixed(zex)})
        F = F - L

        Jp = self.make_preconditioner(TestFunction(Z), TrialFunction(Z))
        problem = NonlinearVariationalProblem(F, z, bcs=bcs, Jp=Jp)
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix="")
        return solver

    def error_norms(self, uh, uex, ph, pex, a=None):
        """Computes the standard and stress error norms
        """
        hypot = lambda x, y: (x**2 + y**2)**0.5
        err_standard = float(hypot(norm(uh-uex, 'H1'), norm(ph-pex)) / hypot(norm(uex, 'H1'), norm(pex)))
        err_stress = float(norm(self.stress(uh, ph) - self.stress(uex, pex), 'L2') / norm(self.stress(uex, pex), 'L2'))
        return err_standard, err_stress

    def solve_mms(self, mesh, degree, solver_parameters=None):
        """Uses the method of manufactured solutions to setup and solve a
           variational problem and returns the error norms.
        """
        _, level = get_level(mesh)
        
        # Construct an exact solution
        zex = self.exact_solution(mesh)

        # Construct the discrete function space
        Z = self.make_function_space(mesh, degree)
        try:
            solver = self.solvers[Z]
        except KeyError:
            quadrature_degree = 4*degree
            make_solver = self.make_nonlinear_solver if self.nonlinear else self.make_linearized_solver
            solver = make_solver(Z, zex, solver_parameters=solver_parameters, quadrature_degree=quadrature_degree)
            self.solvers[Z] = solver

        # Solve the variational problem, and store the solution in uh
        a = solver._problem.J
        z = solver._problem.u
        if not self.nonlinear:
            # Always use zero as the initial guess
            z.assign(0)
        solver.solve()
        uh, ph = z.subfunctions
        uex, pex = zex
        if not self.nonlinear:
            # correct the pressure mean value
            vol = assemble(1*dx(domain=mesh))
            p0 = (1/vol)*assemble((pex-ph)*dx)
            ph.assign(ph + p0)

        dofs = Z.dim()
        err_standard, err_stress = self.error_norms(uh, uex, ph, pex)

        uh.rename("displacement")
        ph.rename("pressure")
        VTKFile(f"output/solution_nonlinear{degree}_{level}.pvd").write(uh, ph)
        return err_standard, err_stress, dofs


class LinearElasticitySolver(HyperElasticitySolver):

    def stress(self, u, p):
        """Computes the stress for linear elasticity
        """
        mesh = as_domain(u)
        mu, rlam = self.material_parameters(mesh)
        d = mesh.geometric_dimension()
        sigma = (2 * mu) * eps(u) + p * Identity(d)
        return sigma

    def energy(self, z, quadrature_degree=None):
        """Computes the energy for linear elasticity
        """
        mesh = as_domain(z)
        mu, rlam = self.material_parameters(mesh)
        try:
            u, p = split(z)
        except ValueError:
            u = z
            p = (1/rlam) * div(u)
        divu = div(u)
        epsu = eps(u)
        W = mu * inner(epsu, epsu) + inner(divu, p) - (rlam/2) * inner(p, p)
        U = W * dx(degree=quadrature_degree)
        return U


class NeoHookeanSolver(HyperElasticitySolver):

    def stress(self, u, p):
        """Computes the stress for neohookean hyperelasticity
        """
        mesh = as_domain(u)
        dim = mesh.geometric_dimension()
        mu, rlam = self.material_parameters(mesh)

        I = Identity(dim)
        F = grad(u) + I
        J = det(F)
        B = dot(F, F.T)
        sigma = (mu/J) * (B - I) + p * I
        return sigma

    def energy(self, z, quadrature_degree=None):
        """Computes the energy for neohookean hyperelasticity
        """
        mesh = as_domain(z)
        dim = mesh.geometric_dimension()
        mu, rlam = self.material_parameters(mesh)
        u, p = split(z)

        # F = deformation gradient
        # J = determinant of F
        I = Identity(dim)
        F = grad(u) + I
        J = det(F)

        logJ = ln(J*J)/2
        trE = inner(F, F) - dim

        W = (mu/2) * (trE - 2*logJ) + inner(J-1, p) - (rlam/2) * inner(p, p)
        U = W * dx(degree=quadrature_degree)
        return U


class RandomDGCoefficientSolver(LinearElasticitySolver):
    """Solve linear elasticity with material parameters that depend on a random coarse function
    """
    def material_parameters(self, mesh):
        # Get constant parameters based on E and nu
        mu_const, rlam_const = LinearElasticitySolver.material_parameters(self, mesh)

        if not hasattr(self, "dg_hierarchy"):
            # Create a new random function on the coarse mesh
            # This function will be the same accross different polynomial degrees and levels of refinement
            DGp = FunctionSpace(mesh, "DG", self.params[2])
            rg = RandomGenerator(PCG64(seed=0))
            r = rg.uniform(DGp, -1.0, 1.0)
            self.dg_hierarchy = FunctionHierarchy(r)

        # Transfer the DG function onto the current mesh
        r = self.dg_hierarchy.get_function(mesh)

        # Include the dependence on the DG function
        mu = mu_const
        rlam = mu_const + (rlam_const - mu_const) * (r ** 2)
        return mu, rlam