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

from math import sqrt

import numpy as np
from OGPIT_hetGPy import OGPIT
import sys, os

sys.path.append("/home/jmlarson/FES_SciDAC/SAD_code/test/")

from make_pyCUTEst_problem_list import make_pyCUTEst_problem_list

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


problems = make_pyCUTEst_problem_list()
nprob = len(problems)

if not os.path.exists("./benchmark_results"):
    os.makedirs("./benchmark_results")

for p in range(nprob):
    problem = problems[p]

    def objfun(x, ns=1):
        val = problem.obj(x)
        # val = val + np.random.normal(0, 1 / sqrt(ns))
        return val

    x0 = problem.x0

    d = len(x0)  # Number of variables [2]
    if d > 10:
        continue
    nfmax = 1e5  # 500  # Maximum number of function evaluations per local minimization [60]
    maxrep = 5e2  # Maximum number of evaluation per iteration (multiplied by ns)
    ninit = max(10, 2 * d)  # Number of initial design points
    nmpmax = 2 * d + 1  # Maximum number of model points [2*n+1]
    gammam = 0.8
    gammap = 0.9
    beta = 1e-3
    eta1 = 0.2
    relvarxnew = 4
    trace = 1
    deter = False
    modtype = "homGP"
    ncand = "small"
    nn = 5 * d
    minnn = d + 1
    maxnn = 200
    delta = 0.2  # initial trust region radius
    mindelta = 1e-6
    maxdelta = 0.5  # maximum trust region radius
    mintheta = min(0.5, 0.1 * sqrt(d))
    maxtheta = min(5 * sqrt(d), 10)
    vredthrestot = 0.3
    iso = False
    normalize = True
    toldist = 1e-4
    boots = True

    # FIX THE STATE OF THE RANDOM NUMBER GENERATORS TO REPRODUCE RESULTS=======
    randstate = 3
    np.random.seed(randstate)

    # Test problem definition
    funcname = objfun
    functruename = objfun
    Low = -np.ones(d)  # np.zeros(d) # 1-by-n Vector of lower bounds [zeros(1,n)]
    Upp = np.ones(d)  # 1-by-n Vector of upper bounds [ones(1,n)]
    xstars = np.atleast_2d(np.zeros(d))  # np.array([[0.1233668, 0.8192679], [0.5421332, 0.1518330], [0.9608988, 0.1640326]])

    # for noise in [0, 0.0001, 0.001, 0.01, 0.1, 1]:
    for noise in [0]:

        task_id = p
        print("task", task_id)

        # Only process if the task_id modulo size is equal to rank
        if task_id % size != rank:
            continue

        outfilename = f"probname={problem.name}_nfmax={nfmax}_noise={noise}.npy"

        if os.path.exists("./benchmark_results/" + outfilename):
            print("Already solved: ", outfilename)
            continue

        if noise == 0:
            deter = True
        else:
            deter = False

        res = OGPIT(
            func=funcname,
            Low=Low,
            Upp=Upp,
            nfmax=nfmax,
            delta=delta,
            mindelta=mindelta,
            maxdelta=maxdelta,
            ninit=ninit,
            trace=trace,
            mintheta=mintheta,
            maxtheta=maxtheta,
            vredthrestot=vredthrestot,
            maxrep=maxrep,
            beta=beta,
            eta1=eta1,
            deter=deter,
            gammam=gammam,
            maxnn=maxnn,
            minnn=minnn,
            relvarxnew=relvarxnew,
            modtype=modtype,
            iso=iso,
            ncand=ncand,
            normalize=normalize,
            toldist=toldist,
            boots=boots,
            lightreturn=False,
        )

        res["grad_after"] = np.zeros_like(res["X"])

        for i in range(len(res["X"])):
            res["grad_after"][i] = problem.grad(res["X"][i])

        np.save("./benchmark_results/" + outfilename, res)
