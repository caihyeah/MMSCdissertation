from firedrake import *
from firedrake.output import VTKFile
from firedrake.mg.utils import get_level
from ufl.domain import as_domain
from ufl import replace, conditional, And
from firedrake import PETSc   
import numpy


class FunctionHierarchy():
    """Class to transfer Functions across levels of a mesh hierarchy.
    """
    tm = TransferManager()

    def __init__(self, u):
        mesh = as_domain(u)
        mh, l = get_level(mesh)
        self.u = {l: u,}

    def get_function(self, mesh):
        mh, lf = get_level(mesh)
        lc = min(self.u)
        lc = max(lf-1, lc)
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
            self.tm.prolong(uc, uf)
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
    def __init__(self, *params, nonlinear=False, exact = "sin"):
        self.params = {}
        for p in params:
            self.params[p.name] = p
        self.solutions = {}
        self.solvers = {}
        self.nonlinear = nonlinear
        self.exact = exact

    def energy(self, u):
        raise NotImplementedError("Please implement the energy")

    def stress(self, u):
        raise NotImplementedError("Please implement the stress")

    def get_parameter(self, param_name):
        try:
            return self.params[param_name]
        except KeyError:
            param = Constant(1.0, name=param_name)
            self.params[param_name] = param
            return param

    def set_parameter(self, param_name, value):
        p = self.get_parameter(param_name)
        p.assign(value)

    def material_parameters(self, mesh):
        # Lame parameters
        nu = self.get_parameter("nu")
        E = self.get_parameter("E")
        mu = E / (2 * (1 + nu))
        rlam = ((1 + nu) * (1 - 2*nu)) / (E*nu)
        return mu, rlam

    def exact_solution(self, mesh):
        """Returns the symbolic representation of the exact solution.
        """
        mu, rlam = self.material_parameters(mesh)
        # Symbolic mesh coordinates
        x = SpatialCoordinate(mesh)
        # incompressible displacement
        A = Constant(1E-2)
        # pressure on the incompressible region
        p0 = cos(x[0]*2*pi) * cos(x[1]*2*pi)
        pex = conditional(abs(rlam) < 1E-12, p0, 0)
        if self.exact == "sin":
            uex = curl(A * sin(x[0]*pi)**2 * sin(x[1]*pi)**2)
        elif self.exact == "exp":
            uex = curl(A * exp(x[0]-x[1]))
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
        # subs = ["on_boundary"]  # impose the BC on the entire boundary
        subs = [1, 2] # impose the BC on vertical edges
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

        gamma = self.get_parameter("gamma")
        rrho = rlam + 1/gamma
        rho = 1/rrho
        aP = (((2*mu)*inner(eps(u), eps(v)) + rho*inner(div(u), div(v)))*dx
              + rrho*inner(p, q)*dx)
        return aP

    def solution(self, Z):
        """Returns the discrete solution for a given mesh and degree.
        If the solution has not been computed, we interpolate the solution
        from the highest degree or finest mesh available, in that order.
        """
        mesh = Z.mesh()
        element = Z.ufl_element()
        degree = element.degree()
        try:
            z = self.solutions[degree].get_function(mesh)
        except KeyError:
            z = Function(Z)
            if len(self.solutions) > 0:
                # Initialize the solution with the maximum degree available on this mesh
                k = max(self.solutions)
                w = self.solutions[k].get_function(mesh)
                for zsub, wsub in zip(z.subfunctions, w.subfunctions):
                    zsub.interpolate(wsub)
            self.solutions[degree] = FunctionHierarchy(z)
        return z

    def make_linearized_solver(self, Z, zex, solver_parameters=None, quadrature_degree=None):
        """Returns a solver for the linearized hyperelasticity problem
        """
        uex, pex = zex
        bcs = self.make_bcs(Z, ubc=uex)
        # Create a discrete Function uh, initialized to 0
        z = self.solution(Z)

        #linearization point equal the interpolation of exact solution
        z0 = Function(Z) 
        for zsub, ex in zip(z.subfunctions, zex):
            zsub.interpolate(ex)
        a = self.make_bilinear_form(z0, quadrature_degree=quadrature_degree)
        test, trial = a.arguments()
        L = a(test, as_mixed(zex))

        aP = None
        if solver_parameters is not None and "augmented_lagrangian" in solver_parameters:
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
        z = self.solution(Z)
        F = self.make_nonlinear_residual(z, quadrature_degree=quadrature_degree)
        L = replace(F, {z: as_mixed(zex)})
        F = F - L

        Jp = None
        if solver_parameters is not None and "augmented_lagrangian" in solver_parameters:
            Jp = self.make_preconditioner(TestFunction(Z), TrialFunction(Z))
        problem = NonlinearVariationalProblem(F, z, bcs=bcs, Jp=Jp)
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix="")
        
        residuals = []
        def _monitor(snes, its, rnorm):
            residuals.append(rnorm)
        solver.snes.setMonitor(_monitor)
        
        return solver,residuals 

    def error_norms(self, uh, uex, ph, pex, a=None):
        """Computes the displacement and pressure error norms
        """
        Lp = "L4" if self.nonlinear else "L2"
        norm_pex = norm(pex, Lp)
        if norm_pex < 1e-14:
            norm_pex = 1
        err_p = float(norm(ph-pex, Lp) / norm_pex)
        if self.nonlinear:
            err_u = float((norm(uh-uex, 'L4')+ norm(grad(uh-uex), 'L4')) / (norm(uex, 'L4')+norm(grad(uex), 'L4')))
            err_stress = float(norm(self.stress(uh, ph) - self.stress(uex, pex), 'L4') / norm(self.stress(uex, pex), 'L4'))
            err_incomp = float(norm(det(grad(uh))-det(grad(uex)),'L2'))
            return err_u, err_p, err_stress,err_incomp
        else:
            err_u = float(norm(uh-uex, 'H1') / norm(uex, 'H1'))
            return err_u, err_p

    def solve_mms(self, mesh, degree, param_ranges=None, solver_parameters=None):
        """Uses the method of manufactured solutions to setup and solve a
           variational problem and returns the error norms.
           :kwarg param_ranges: a dictionary of names and values defining the steps
                                of a continuation strategy.
        """
        _, level = get_level(mesh)

        if param_ranges is None:
            param_ranges = {}
        # Construct an exact solution
        zex = self.exact_solution(mesh)
        uex, pex = zex

        # Construct the discrete function space
        Z = self.make_function_space(mesh, degree)
        try:
            solver = self.solvers[Z]
        except KeyError:
            quadrature_degree = 4*degree
            make_solver = self.make_nonlinear_solver if self.nonlinear else self.make_linearized_solver
            solver,residuals = make_solver(Z, zex, solver_parameters=solver_parameters, quadrature_degree=quadrature_degree)
            self.solvers[Z] = (solver,residuals)

        solver, residuals = self.solvers[Z]
        a = solver._problem.J
        z = solver._problem.u
        uh, ph = z.subfunctions

        def step():
            # Solve the variational problem, and store the solution in uh
            # print the errors and return parameter values and errors
            solver.solve()

            ''' No longer needed because bcs \neq "on_boudary"
            if not self.nonlinear:
                # Correct the pressure mean value
                vol = assemble(1*dx(domain=mesh))
                p0 = (1/vol)*assemble((pex-ph)*dx)
                ph.interpolate(ph + p0)
            '''
            if self.nonlinear:
                import numpy as np, pathlib
                pathlib.Path("output").mkdir(exist_ok=True)
                fname = f"output/residual_deg{degree}_lev{level}_{self.exact}.txt"
                np.savetxt(fname, residuals, fmt="%.8e")


            pvals = [float(self.get_parameter(pname)) for pname in self.params]
            errors = self.error_norms(uh, uex, ph, pex, a=a)
            print("errors", errors)
        


            return pvals, errors

        # dict of iterators for continuation
        param_iter = {pname: iter(pvals) for pname, pvals in param_ranges.items()}

        errors_list = []

        # Initialize the parameters
        for pname in param_iter:
            pval = next(param_iter[pname])
            print(f"Setting initial value {pname}={pval}")
            self.set_parameter(pname, pval)

        # First continuation step
        result = step()
        errors_list.append(result)

        # Rest of the continuation steps
        for pname in param_iter:
            for pnew in param_iter[pname]:
                pold = float(self.get_parameter(pname))
                pmid = pnew
                its = 0
                while True:
                    print(f"{its} Continuation step {pname}={pmid}")
                    its = its + 1
                    self.set_parameter(pname, pmid)
                    try:
                        result = step()
                        if pmid == pnew:
                            break
                        pold, pmid = pmid, pnew
                    except ConvergenceError as e:
                        print(e)
                        if its > 200:
                            raise ConvergenceError("Continuation failed to converge after {its} steps.")
                        pmid = (pold + pmid)/2
                errors_list.append(result)
        #import matplotlib.pyplot as plt
        #trisurf(uh)
        #plt.show()

        uh.rename("displacement")
        ph.rename("pressure")
        VTKFile(f"output/solution{degree}_{level}.pvd").write(uh, ph)

        dofs = Z.dim()
        return errors_list, dofs


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
    
    def error_norms(self, uh, uex, ph, pex, a=None):
        """Computes the standard and stress error norms
        """
        hypot = lambda x, y: (x**2 + y**2)**0.5
        err_standard = float(hypot(norm(uh-uex, 'H1'), norm(ph-pex)) / hypot(norm(uex, 'H1'), norm(pex)))
        err_stress = float(norm(self.stress(uh, ph) - self.stress(uex, pex), 'L2') / norm(self.stress(uex, pex), 'L2'))
        return err_standard, err_stress


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

        W = (mu/2) * (trE) + inner(J-1, p)
        #W = (mu/2) * (trE - 2*logJ) + inner(J-1, p) - (rlam/2) * inner(p, p)
        U = W * dx(degree=quadrature_degree)
        return U


class HolzapfelOgdenSolver(HyperElasticitySolver):
    def stress(self, u, p):
        """Compute the stress for the Holzapfel-Ogden hyperelastic model."""
        mesh = as_domain(u)
        dim = mesh.geometric_dimension()

        # Material parameters
        a = 0.496
        b = 0.041
        af = 0.193
        bf = 0.176
        as_ = 0.123
        bs = 0.209
        afs = 0.162
        bfs = 0.166

        I = Identity(dim)
        F = grad(u) + I  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green deformation tensor
        J = det(F)  # Jacobian determinant
        
        # Calculate I1(C)
        I1 = inner(F,F)
        
        # Calculate I4_f(C) and I4_s(C)
        f0 = as_vector([1, 0])  # f_0 vector
        s0 = as_vector([0, -1]) # s_0 vector
        I4_f = inner(f0, C * f0)
        I4_s = inner(s0, C * s0)
        
        # Calculate I8_fs(C)
        I8_fs = inner(f0, C * s0)

        # Define the ()_+ operation
        def pos_part(x):
            return (x + abs(x)) / 2

        # Manually computed derivative of Psi with respect to F
        Psi_F = a * exp(b * (I1 - dim)) * F \
            + af * exp(bf * pos_part(I4_f - 1)**2) * pos_part(I4_f - 1) *  (2 * F * outer(f0, f0)) \
            + as_ * exp(bs * pos_part(I4_s - 1)**2) * pos_part(I4_s - 1) * (2 * F * outer(s0, s0)) \
            + afs * exp(bfs * I8_fs**2) * I8_fs * (F * 2 * (outer(f0, s0)))

        # Compute the stress
        sigma = (1 / J) * F * Psi_F + p * I  # Compute the Piola-Kirchhoff stress tensor
        return sigma

    def energy(self, z, quadrature_degree=None):
        """Compute the energy for the Holzapfel-Ogden hyperelastic model."""
        mesh = as_domain(z)
        dim = mesh.geometric_dimension()
        u, p = split(z)

        # Material parameters
        a = 0.496
        b = 0.041
        af = 0.193
        bf = 0.176
        as_ = 0.123
        bs = 0.209
        afs = 0.162
        bfs = 0.166

        I = Identity(dim)
        F = grad(u) + I  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green deformation tensor
        J = det(F)  # Jacobian determinant
        
        # Calculate I1(C)
        I1 = tr(C)
        
        # Calculate I4_f(C) and I4_s(C)
        f0 = as_vector([1, 0])  # f_0 vector
        s0 = as_vector([0, -1]) # s_0 vector
        I4_f = inner(f0, C * f0)
        I4_s = inner(s0, C * s0)
        
        # Calculate I8_fs(C)
        I8_fs = inner(f0, C * s0)

        # Define the ()_+ operation
        def pos_part(x):
            return (x + abs(x)) / 2

        # Construct the Holzapfel-Ogden energy density function
        W = (a / (2 * b)) * exp(b * (I1 - dim)) \
            + (afs / (2 * bfs)) * (exp(bfs * I8_fs**2) - 1) \
            + (af / (2 * bf)) * (exp(bf * pos_part(I4_f - 1)**2) - 1) \
            + (as_ / (2 * bs)) * (exp(bs * pos_part(I4_s - 1)**2) - 1)
        
        # Define the total energy expression
        U = W * dx(degree=quadrature_degree)
        return U

class OgdenSolver(HyperElasticitySolver):
    def stress(self, u, p):
        mesh = as_domain(u)
        dim = mesh.geometric_dimension()
        mu = self.material_parameters(mesh)
        alpha = self.get_parameter("alpha")

        I = Identity(dim)
        F = grad(u) + I
        C = F.T * F
        J = det(F)
        I1 = tr(C)

        W = (mu / alpha) * (I1 ** (alpha / 2) - 3.0)
        P = ufl.derivative(W, F) + p * inv(F).T
        return P

    def energy(self, z, quadrature_degree=None):
        mesh = as_domain(z)
        dim = mesh.geometric_dimension()
        u, p = split(z)
        mu = self.material_parameters(mesh)
        alpha = self.get_parameter("alpha")

        I = Identity(dim)
        F = grad(u) + I
        C = F.T * F
        J = det(F)
        I1 = tr(C)

        W = (mu / alpha) * (I1 ** (alpha / 2) - 3.0)
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
