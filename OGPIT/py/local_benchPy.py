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

usr = "MB" # "JL"
if usr == "MB":
    sys.path.append("/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python")
    sys.path.append(".")

from OGPIT_hetGPy import OGPIT

from scipy.interpolate import make_interp_spline,interp1d

# Standard branin
def branin(x):
    x = np.atleast_2d(x)
    low = np.array([-5, 0])
    upp = np.array([10, 15])
    x = low + x * (upp - low)
    a = 1
    b = 5 / (4 * pi**2)
    c = 5 / pi
    d = 6
    e = 10
    f9 = 1 / (8 * pi)
    f = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - d) ** 2 + e * (1 - f9) * cos(x[:, 0]) + e
    return f

def jeffsfav(x):
    f = np.sum(x**2)
    return f

def quarf(x):
    f = np.sum(x**4)
    return f

# Rosenbrock 2d on [-1.5, 1.5]
def rosen(x):
    xx = x * 3 - 1.5
    xi = xx[0]
    xnext = xx[1]
    f = np.sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2)
    return f

# Normalized Rosenbrock 4d on [-1.5, 1.5]
def rosen2(x):
    m = 382658.057227524
    s = 375264.858362295
    x = 15 * x - 5
    x1 = x[0:2]
    x2 = x[1:3]
    f = np.sum(100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)
    f = (f - m) / s
    return f

def noisyfun(fun, x, noise, ns):
    f = fun(x)
    f = f + np.random.normal(loc=0, scale=noise / sqrt(ns))
    return f

if __name__ == "__main__":
    if not os.path.exists("./benchmark_results"):
        os.makedirs("./benchmark_results")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("Local bench", flush=True)
    # exp_name = "OGPIT_local_test_benchPy"
    # opt_Problems = np.arange(9) + 1
    opt_Problems = [1,2,3,4,5,6,8,9]

    prob_count = 0
    for prob in opt_Problems:
        print(prob, flush=True)

        random.seed(int(prob))

        nrep = 10 # 100
        if prob == 1:
            d = 2
            fn = jeffsfav
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 2:
            d = 4
            fn = jeffsfav
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 3:
            d = 6
            fn = jeffsfav
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 4:
            d = 2
            fn = quarf
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 5:
            d = 4
            fn = quarf
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 6:
            d = 6
            fn = quarf
            xstars = np.zeros([1,d])
            fstar = 0.
            lower = -np.ones(d)
            upper = np.ones(d)

        if prob == 7:
            d = 2
            fn = branin
            xstars = np.zeros([1,d])
            xstars[0,0] = 0.9616520
            xstars[0,1] = 0.15
            fstar = branin(xstars)
            lower = np.zeros(d)
            upper = np.ones(d)

        if prob == 8:
            d = 2
            fn = rosen
            xstars = (np.ones([1,d]) + 1.5)/3
            fstar = 0
            lower = np.zeros(d)
            upper = np.ones(d)

        if prob == 9:
            d = 4
            fn = rosen2
            fntrue = rosen2
            xstars = 0.4 * np.ones([1,d])
            fstar = 0
            lower = np.zeros(d)
            upper = np.ones(d)

        # if prob % size != rank:
        #     continue

        budget = int(1e4*(d+1))  #5e4 # 1e5 #1e5 #500000 #100000
        ninit = max(10, 2 * d)

        gammam = 0.8
        gammap = 0.5

        beta = 1e-3
        eta1 = 0.2

        nn = 5 * d
        minnnTR = d + 1  # d + 1 is better for deterministic
        maxnn = max(200, 10 * d)
        delta = 0.1  # initial trust region radius
        mindelta = 1e-6
        maxdelta = 0.5  # maximum trust region radius
        mintheta = None  # min(0.5, 0.1 * sqrt(d))
        maxtheta = None  # min(5 * sqrt(d), 10)
        vredthrestot = 0.3
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

        # noises = np.array((0, 0.001, 0.01, 0.1))  # Noise std
        noises = np.array((0, 0.001, 0.01, 0.1))  # Noise std

        all_res = np.ones((noises.shape[0], nrep, 220))
        nid = 0 # noise id
        for nois in noises:
            print(nois)

            prob_count += 1

            xps = np.sort(np.concatenate((np.linspace(10,90,9),
                                          np.linspace(100,1000,10),
                                          np.linspace(0,budget,201))))
            xps[0] = 1
            if nois == 0:
                deter = True
                budget0 = int(1e3*(d+1))
            else:
                deter = False
                budget0 = budget

            for ii in np.arange(nrep):
                if prob_count % size != rank:
                    continue

                random.seed(int(ii))

                outfilename1 = f"pydefault_probname={prob}_nfmax={budget}_noise={nois}_seed={ii}_regret_and_naiveregret.npy"

                if os.path.exists("./benchmark_results/" + outfilename1):
                    print("Already solved: ", outfilename1)
                    continue

                def func(x, ns=1):
                    global nois
                    global fn
                    return noisyfun(fn, x, noise = nois, ns=ns)

                starttime = time.time()
                res = OGPIT(func=func, Low=lower, Upp=upper, nfmax=budget0, gammam=gammam, delta=delta, mindelta=mindelta, maxdelta=maxdelta,
                            mintheta=mintheta, maxtheta=maxtheta, trace=trace, vredthrestot=vredthrestot, maxrep=maxrep, beta=beta, eta1=eta1,
                            minnn=minnn, maxnn=maxnn, deter=deter, modtype=model_type, iso=iso, normalize=True, ninit=ninit, acqtype=acq_type,
                            imsevar = imsevar, lightreturn=lightreturn, boots=True)
                runtime = time.time() - starttime

                # Naive regret: compute regret at evaluated points
                naiveregret = np.ones(min(res["Xall"].shape[0],int(budget0)))
                for j in np.arange(naiveregret.size):
                    naiveregret[j] = fn(res["Xall"][j,:]) - np.min(fstar)

                # Realistic regret: compute regret based on the current best estimate (i.e., the center at the considered iteration)
                regret = np.ones(res["Xks"].shape[0])
                for j in np.arange(regret.size):
                    regret[j] = fn(res["Xks"][j,:]) - np.min(fstar)

                stpf = interp1d(res["evalits"], regret, kind="previous", fill_value="extrapolate")
                regretatxps = stpf(xps)
                all_res[nid, ii, :] = regretatxps

                # plt.plot(xps, np.log10(regretatxps))
                # plt.plot(np.arange(naiveregret.size)+1, np.log10(naiveregret))
                # plt.show(block=True)
                np.save("./benchmark_results/" + outfilename1, {'regret': regret, 'regretatxps': regretatxps,
                                                                'naiveregret': naiveregret, 'evalits':res['evalits'], 'time':runtime}) #'X': res['X']})
            nid = nid + 1

    # Postprocessing
