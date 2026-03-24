"""
MIT License

Stochastic Derivative-Free Optimization (StochDFO)
Part of POptUS: Practical Optimization Using Structure
Copyright (c) 2026, Mickaël Binois and UChicago Argonne LLC through Argonne
National Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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

test_cma = False
test_cman = False

from OGPIT_hetGPy import OGPIT
from calfun import calfun
from dfoxs import dfoxs
from pathlib import Path

def log_and_abort(msg):
    print()
    print(msg)
    print()
    sys.exit(1)


if __name__ == "__main__":
    # Setup use of BenDFO clone based on user-provided path
    if "BENDFO_PATH" not in os.environ:
        log_and_abort(
            "Please set the BENDFO_PATH environment variable to root of a BenDFO clone"
        )
    bendfo_path = Path(os.environ["BENDFO_PATH"]).resolve()
    if not bendfo_path.is_dir():
        log_and_abort(
            "Invalid path specified in BENDFO_PATH environment variable"
        )
    sys.path.append(bendfo_path.joinpath("py"))

    if not os.path.exists("./benchmark_results"):
        os.makedirs("./benchmark_results")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    factor = 10
    probs = np.loadtxt(bendfo_path.joinpath("data", "dfo.dat"))
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
                else:
                    deter = False
                    budget0 = budget

                xps = np.sort(np.concatenate((np.linspace(10, 90, 9),
                                              np.linspace(100, 1000, 10),
                                              np.linspace(0, budget0, 201))))
                xps[0] = 1

                random.seed(int(rep))
                if test_cma:
                    outfilename1 = f"cma_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"
                elif test_cman:
                    outfilename1 = f"cman_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"
                else:
                    outfilename1 = f"pydefault_bendfo_row={bendfo_row}_nfmax={budget}_noise={nois}_seed={rep}_regret_and_naiveregret.npy"

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
                if test_cma:
                    allXs = np.copy(X_0)
                    def objective2(y, ns=1):
                        global allXs
                        allXs = np.vstack((allXs, y))
                        return objective(y, ns)

                    xopt, es =cma.fmin2(objective2, X_0, delta, {'bounds': [lower, upper], 'maxfevals': budget0},noise_handler=False)
                    # compute centers for Xks
                    psize = int(4 + np.floor(3*np.log(d)))
                    allXks = np.zeros([int(np.floor(allXs.shape[0] / psize)), d])
                    for ii in np.arange(allXks.shape[0]):
                        allXks[ii,:] = np.mean(allXs[np.arange(ii*psize+1,(ii+1)*psize+1),:], axis=0)

                    res = dict(Xall= allXs, Xks= allXks, evalits= np.arange(1,allXks.shape[0]+1)*psize)
                elif test_cman:
                    allXs = np.copy(X_0)
                    def objective2(y, ns=1):
                        global allXs
                        allXs = np.vstack((allXs, y))
                        return objective(y, ns)

                    xopt, es =cma.fmin2(objective2, X_0, delta, {'bounds': [lower, upper], 'maxfevals': budget0},noise_handler=True)
                    res = dict(Xall= allXs, Xks= allXs, evalits= np.arange(0, allXs.shape[0])+1)
                else:
                    res = OGPIT(func=objective, Low=lower, Upp=upper, nfmax=budget0, delta=delta, mindelta=mindelta, maxnn=maxnn,
                        maxdelta=maxdelta, ninit=ninit, nn=nn, trace=trace, mintheta=mintheta, maxtheta=maxtheta, acqtype=acq_type,
                        vredthrestot=vredthrestot, maxrep=maxrep, beta=beta, eta1=eta1, deter=deter, normalize=True, imsevar = imsevar,
                        gammam=gammam, gammap=gammap, lightreturn=lightreturn, minnnTR=minnnTR, modtype=model_type, boots=True, iso=iso,
                        Xinit=Xinit,Zinit=Zinit)
                runtime = time.time() - starttime

                # Check initial point is present
                # if not all(res['Xall'][0] == X_0):
                if np.linalg.norm(res['Xall'][0] - X_0) >= 1e-10:
                    print(X_0)
                    print(res['Xall'][0])
                    print(res['Xall'][0] == X_0)
                    print(np.linalg.norm(res['Xall'][0] - X_0))
                    sys.exit("Problem with using initial point" + outfilename1)

                # Naive regret: compute regret at evaluated points
                naiveregret = np.ones(min(res["Xall"].shape[0],int(budget0)))
                for j in np.arange(naiveregret.size):
                    naiveregret[j] = fn(res["Xall"][j,:]) # - np.min(fstar)

                regret = np.ones(res["Xks"].shape[0])
                for j in np.arange(regret.size):
                    regret[j] = fn(res["Xks"][j,:]) - np.min(fstar)

                stpf = interp1d(res["evalits"], regret, kind="previous", fill_value="extrapolate")
                regretatxps = stpf(xps)

                np.save("./benchmark_results/" + outfilename1, {'regret': regret, 'regretatxps': regretatxps,
                                                                    'naiveregret': naiveregret, 'evalits':res['evalits'], 'time':runtime}) #'X': res['X']})


