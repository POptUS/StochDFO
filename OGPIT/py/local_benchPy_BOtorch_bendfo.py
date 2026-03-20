from __future__ import division

import sys, os
from math import cos, exp, pi, sqrt

import hetgpy as hgp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import time

from mpi4py import MPI

from scipy.interpolate import make_interp_spline,interp1d
from scipy import optimize, spatial
import cma

import math
import warnings
from dataclasses import dataclass

## Based on https://botorch.org/docs/tutorials/turbo_1/
import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qNoisyExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

test_Turbo = True
test_botorch = False
test_logEI = False

# maxtime = 3600 # Stop after 1h of running time
maxtime = 600 # Stop after 1h of running time

# usr = "MB" # "JL"
usr = "JL" # "JL"
if usr == "MB":
    # sys.path.append("/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python/")
    sys.path.append("/home/mbinois/Documents/GitProjects/BenDFO/py/")
    # sys.path.append(".")

# from OGPIT_hetGPy import OGPIT3
from calfun import calfun
from dfoxs import dfoxs


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 1e-10
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


if __name__ == "__main__":
    if not os.path.exists("./benchmark_results"):
        os.makedirs("./benchmark_results")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    factor = 10
    if usr == "MB":
        probs = np.loadtxt("/home/mbinois/Documents/GitProjects/BenDFO/data/dfo.dat")
    else:
        probs = np.loadtxt("/home/jmlarson/research/poptus/BenDFO/data/dfo.dat")

    probtype = 'smooth'
    nreps = 30
    noises = np.array((0, 0.001, 0.1, 10))  # Noise std
    
    gammam = 0.8
    gammap = 0.5
    beta = 1e-3
    eta1 = 0.2
    delta = 0.1  # initial trust region radius
    mindelta = 1e-10
    maxdelta = 0.5  # maximum trust region radius
    mintheta = None  # min(0.5, 0.1 * sqrt(d))
    maxtheta = None  # min(5 * sqrt(d), 10)
    vredthrestot = 0.2
    maxrep = 5e2
    # testpnois <- True
    acq_type = "EI"  # "EI" or "LineSearch" or "qRI"
    model_type = "homGP"
    iso = False
    trace = 0
    lightreturn = False
    plot = False
    boots = True
    verbose = False
    imsevar = 10

    prob_count = 0
    for bendfo_row, (nprob, d, m, factor_power) in enumerate(probs):
        d = int(d)
        m = int(m)
        nprob = int(nprob)

        if d > 9 or nprob == 10 or nprob == 12 or nprob == 13 or nprob == 17:
            continue

        budget = int(1e4*(d+1))  #5e4 # 1e5 #1e5 #500000 #100000
        ninit = max(10, 2 * d)

        nn = 5 * d
        minnnTR = d + 1  # d + 1 is better for deterministic
        maxnn = max(200, 10 * d)

        X_0 = dfoxs(d, nprob, int(factor**factor_power))

        lower = -5000 * np.ones(d)  # 1-by-n Vector of lower bounds [zeros(1,n)]
        upper = 5000 * np.ones(d)  # 1-by-n Vector of upper bounds [ones(1,n)]


        def fn0(y):
            out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
            return np.squeeze(out)

        res_ref = optimize.minimize(
            fun=fn0,
            x0=X_0,
            method="L-BFGS-B",
            bounds=[(l, u) for l, u in zip(lower, upper)],
        )

        xoptref, esref = cma.fmin2(fn0, X_0, delta, {'bounds': [lower, upper], 'verbose':-9})
        cmf = dict(esref.result._asdict())['fbest']

        # print(bendfo_row)
        # print(X_0)

        if res_ref['fun'] < cmf:
            fstar = res_ref['fun']
            # print(res_ref['x'])
        else:
            fstar = cmf
            # print(xoptref)

        # print(fstar)
        # print("")

        for rep in range(nreps):
            for ii,nois in enumerate(noises):
                prob_count += 1

                if prob_count % size != rank:
                    continue

                if nois == 0:
                    deter = True
                    budget0 = int(1e3*(d+1)) # Less budget if deterministic
                    upnoise = 1e-3
                else:
                    deter = False
                    budget0 = budget
                    upnoise = 1e2

                xps = np.sort(np.concatenate((np.linspace(10, 90, 9),
                                              np.linspace(100, 1000, 10),
                                              np.linspace(0, budget0, 201))))
                xps[0] = 1

                random.seed(int(rep))
                if test_Turbo:
                    outfilename1 = f"turbo_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"
                elif test_botorch:
                    outfilename1 = f"botorch_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"
                else:
                    outfilename1 = f"logEI_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"

                if os.path.exists("./benchmark_results/" + outfilename1):
                    print("Already solved: ", outfilename1, flush=True)
                    continue

                def objective(y, ns=1):
                        # It is possible to have python use the same objective values via
                        # octave. This can be slow on some systems. To (for example)
                        # test difference between matlab and python, used the following
                        # line and add "from oct2py import octave" on a system with octave
                        # installed.
                        # out = octave.feval("calfun_wrapper", y, m, nprob, "smooth", [], 1, 1)
                    out = calfun(y, m, nprob, "smooth", 0, num_outs=2)[0]
                        # assert len(out) == m, "Incorrect output dimension"

                    out = out + np.random.normal(loc=0, scale=nois / sqrt(ns))
                    return np.squeeze(out)

                def fn(y):
                    out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
                    return np.squeeze(out)


                starttime = time.time()
                Xinit = (np.atleast_2d(X_0) - lower) /(upper - lower)
                Zinit = np.ones(1) * objective(X_0)

                if test_Turbo:
                    # fun = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
                    # fun.bounds[0, :].fill_(-5)
                    # fun.bounds[1, :].fill_(10)
                    # dim = fun.dim
                    # lb, ub = fun.bounds

                    batch_size = 1
                    n_init = ninit
                    max_cholesky_size = float("inf")  # Always use Cholesky

                    state = TurboState(dim=d, batch_size=batch_size)
                    # print(state)

                    # def eval_objective(x):
                    #     """This is a helper function we use to unnormalize and evalaute a point"""
                    #     tmp = fun(unnormalize(x, fun.bounds))
                    #     print(tmp)
                    #     return fun(unnormalize(x, fun.bounds))

                    # allXs = np.copy(X_0)
                    def eval_objective2(y):
                        x = y.numpy()
                        # global allXs
                        global lower
                        global upper
                        # allXs = np.vstack((allXs, x))
                        res = objective(x * (upper - lower) + lower)
                        return -res

                    X_turbo = get_initial_points(d, n_init)
                    X_turbo[0,:] = torch.from_numpy((X_0 - lower) /(upper - lower))
                    Y_turbo = torch.tensor(
                        [eval_objective2(x) for x in X_turbo], dtype=dtype, device=device
                    ).unsqueeze(-1)

                    state = TurboState(d, batch_size=batch_size, best_value=max(Y_turbo).item())

                    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
                    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
                    N_CANDIDATES = min(5000, max(2000, 200 * d)) if not SMOKE_TEST else 4

                    torch.manual_seed(int(rep))
                    keepgo = True

                    while not state.restart_triggered and len(Y_turbo) < budget0 and time.time() - starttime < maxtime and keepgo:  # Run until TuRBO converges
                        try:
                            # Fit a GP model
                            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
                            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, upnoise))
                            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                                MaternKernel(
                                    nu=2.5, ard_num_dims=d, lengthscale_constraint=Interval(0.005, 4.0)
                                )
                            )
                            model = SingleTaskGP(
                                X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
                            )
                            mll = ExactMarginalLogLikelihood(model.likelihood, model)

                            # Do the fitting and acquisition function optimization inside the Cholesky context
                            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                                # Fit the model
                                fit_gpytorch_mll(mll)

                                # Create a batch
                                X_next = generate_batch(
                                    state=state,
                                    model=model,
                                    X=X_turbo,
                                    Y=train_Y,
                                    batch_size=batch_size,
                                    n_candidates=N_CANDIDATES,
                                    num_restarts=NUM_RESTARTS,
                                    raw_samples=RAW_SAMPLES,
                                    acqf="ts",
                                )

                            Y_next = torch.tensor(
                                [eval_objective2(x) for x in X_next], dtype=dtype, device=device
                            ).unsqueeze(-1)

                            # Update state
                            # oldbest = state.best_value
                            state = update_state(state=state, Y_next=Y_next)

                            # Append data
                            X_turbo = torch.cat((X_turbo, X_next), dim=0)
                            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

                            # Print current status
                            print(
                                f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
                            )
                            # if oldbest == state.best_value:
                            #     print("tot")
                        except:
                            keepgo=False
                    res = dict(Xall=X_turbo.numpy()*(upper - lower) + lower, evalits=np.arange(0, X_turbo.shape[0]) + 1)

                if test_botorch:
                    batch_size = 1
                    n_init = ninit

                    def eval_objective2(y):
                        x = y.numpy()
                        # global allXs
                        global lower
                        global upper
                        # allXs = np.vstack((allXs, x))
                        res = objective(x * (upper - lower) + lower)
                        return -res

                    X_ei = get_initial_points(d, n_init)
                    X_ei[0, :] = torch.from_numpy((X_0 - lower) /(upper - lower))
                    Y_ei = torch.tensor(
                        [eval_objective2(x) for x in X_ei], dtype=dtype, device=device
                    ).unsqueeze(-1)

                    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
                    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
                    N_CANDIDATES = min(5000, max(2000, 200 * d)) if not SMOKE_TEST else 4

                    torch.manual_seed(int(rep))
                    keepgo = True

                    while len(Y_ei) < budget0 and time.time() - starttime < maxtime and keepgo:  # Run until budget used or time limit
                        try:
                            # Fit a GP model
                            train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
                            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, upnoise))
                            model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
                            mll = ExactMarginalLogLikelihood(model.likelihood, model)
                            fit_gpytorch_mll(mll)

                            # Create a batch
                            if deter:
                                ei = qExpectedImprovement(model, train_Y.max())
                            else:
                                ei = qNoisyExpectedImprovement(model, X_ei)

                            candidate, acq_value = optimize_acqf(
                                ei,
                                bounds=torch.stack(
                                    [
                                        torch.zeros(d, dtype=dtype, device=device),
                                        torch.ones(d, dtype=dtype, device=device),
                                    ]
                                ),
                                q=batch_size,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                            )

                            Y_next = torch.tensor(
                                [eval_objective2(x) for x in candidate], dtype=dtype, device=device
                            ).unsqueeze(-1)

                            # Append data
                            X_ei = torch.cat((X_ei, candidate), axis=0)
                            Y_ei = torch.cat((Y_ei, Y_next), axis=0)

                            # Print current status
                            print(f"{len(X_ei)}) Best value: {Y_ei.max().item():.2e}")

                            # if oldbest == state.best_value:
                            #     print("tot")
                        except:
                            keepgo=False

                    res = dict(Xall=X_ei.numpy()*(upper - lower) + lower,  evalits=np.arange(0, X_ei.shape[0]) + 1)

                if test_logEI:
                    batch_size = 1
                    n_init = ninit

                    def eval_objective2(y):
                        x = y.numpy()
                        # global allXs
                        global lower
                        global upper
                        # allXs = np.vstack((allXs, x))
                        res = objective(x * (upper - lower) + lower)
                        return -res

                    X_ei = get_initial_points(d, n_init)
                    X_ei[0, :] = torch.from_numpy((X_0 - lower) /(upper - lower))
                    Y_ei = torch.tensor(
                        [eval_objective2(x) for x in X_ei], dtype=dtype, device=device
                    ).unsqueeze(-1)

                    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
                    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
                    N_CANDIDATES = min(5000, max(2000, 200 * d)) if not SMOKE_TEST else 4

                    torch.manual_seed(int(rep))
                    keepgo = True

                    while len(Y_ei) < budget0 and time.time() - starttime < maxtime and keepgo:  # Run until budget used or time limit
                        try:
                            # Fit a GP model
                            train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
                            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, upnoise))
                            model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
                            mll = ExactMarginalLogLikelihood(model.likelihood, model)
                            fit_gpytorch_mll(mll)

                            # Create a batch
                            if deter:
                                ei = qLogExpectedImprovement(model, train_Y.max())
                            else:
                                ei = qLogNoisyExpectedImprovement(model, X_ei)

                            candidate, acq_value = optimize_acqf(
                                ei,
                                bounds=torch.stack(
                                    [
                                        torch.zeros(d, dtype=dtype, device=device),
                                        torch.ones(d, dtype=dtype, device=device),
                                    ]
                                ),
                                q=batch_size,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                            )

                            Y_next = torch.tensor(
                                [eval_objective2(x) for x in candidate], dtype=dtype, device=device
                            ).unsqueeze(-1)

                            # Append data
                            X_ei = torch.cat((X_ei, candidate), axis=0)
                            Y_ei = torch.cat((Y_ei, Y_next), axis=0)

                            # Print current status
                            print(f"{len(X_ei)}) Best value: {Y_ei.max().item():.2e}")

                            # if oldbest == state.best_value:
                            #     print("tot")
                        except:
                            keepgo = False

                    res = dict(Xall=X_ei.numpy() * (upper - lower) + lower,  evalits=np.arange(0, X_ei.shape[0]) + 1)

                # elif test_cman:
                #     allXs = np.copy(X_0)
                #     def objective2(y, ns=1):
                #         global allXs
                #         allXs = np.vstack((allXs, y))
                #         return objective(y, ns)
                #
                #     xopt, es =cma.fmin2(objective2, X_0, delta, {'bounds': [lower, upper], 'maxfevals': budget0},noise_handler=True)
                #     res = dict(Xall= allXs, Xks= allXs, evalits= np.arange(0, allXs.shape[0])+1)

                # else:
                #     res = OGPIT3(func=objective, Low=lower, Upp=upper, nfmax=budget0, delta=delta, mindelta=mindelta, maxnn=maxnn,
                #         maxdelta=maxdelta, ninit=ninit, nn=nn, trace=trace, mintheta=mintheta, maxtheta=maxtheta, acqtype=acq_type,
                #         vredthrestot=vredthrestot, maxrep=maxrep, beta=beta, eta1=eta1, deter=deter, normalize=True, imsevar = imsevar,
                #         gammam=gammam, gammap=gammap, lightreturn=lightreturn, minnnTR=minnnTR, modtype=model_type, boots=True, iso=iso,
                #         Xinit=Xinit,Zinit=Zinit)
                runtime = time.time() - starttime

                # Check initial point is present
                if not all(res['Xall'][0] == X_0):
                    print("Problem with using initial point: X0=", X_0, "while Xall[0,:]=",res['Xall'][0])

                # Naive regret: compute regret at evaluated points
                naiveregret = np.ones(min(res["Xall"].shape[0],int(budget0)))
                for j in np.arange(naiveregret.size):
                    naiveregret[j] = fn(res["Xall"][j,:]) # - np.min(fstar)

                regret = naiveregret - np.min(fstar)

                regretatxps = regret[xps[xps < naiveregret.size].astype("int")-1]

                np.save("./benchmark_results/" + outfilename1, {'regret': regret, 'regretatxps': regretatxps,
                                                                    'naiveregret': naiveregret, 'evalits':res['evalits'], 'time':runtime}) #'X': res['X']})


