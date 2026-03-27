"""
MIT License

Stochastic Derivative-Free Optimization (StochDFO)
Part of POptUS: Practical Optimization Using Structure
Copyright (c) 2026, Inria and UChicago Argonne LLC through Argonne National
Laboratory (subject to receipt of any required approvals from the U.S.  Dept. of
Energy).  All rights reserved.

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

from math import cos, exp, pi, sqrt

import hetgpy as hgp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from hetgpy.auto_bounds import auto_bounds
from hetgpy.covariance_functions import cov_gen, euclidean_dist
from hetgpy.find_reps import find_reps
from hetgpy.IMSE import Wij, mi
from hetgpy.LOO import LOO_preds
from hetgpy.optim import crit_EI, crit_logEI
from hetgpy.qEI import qEI_cpp
from hetgpy.utils import duplicated
from pyDOE import lhs as lhspy
from scipy.linalg.lapack import dtrtri

# import os
# os.environ['R_HOME'] = '/usr/bin/R'


def testfcn3(x, ns=1):
    global noise
    x = np.atleast_2d(x)
    x0 = x
    low = np.array([-5, 0])
    upp = np.array([10, 15])
    x = low + x * (upp - low)
    a = 1
    b = 5.1 / (4 * pi**2)
    c = 5 / pi
    d = 6
    e = 10
    f9 = 1 / (8 * pi)
    f = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - d) ** 2 + e * (1 - f9) * cos(x[:, 0]) + e
    f = f + 0.1 * x[:, 0] + 0.01 * x[:, 1]
    f = f + np.random.normal(0, noise * np.linalg.norm(x0) / sqrt(ns))
    return f


def testfcn3_true(x):
    # global noise, ns
    # x0 = x
    x = np.atleast_2d(x)
    low = np.array([-5, 0])
    upp = np.array([10, 15])
    x = low + x * (upp - low)
    a = 1
    b = 5.1 / (4 * pi**2)
    c = 5 / pi
    d = 6
    e = 10
    f9 = 1 / (8 * pi)
    f = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - d) ** 2 + e * (1 - f9) * cos(x[:, 0]) + e
    f = f + 0.1 * x[:, 0] + 0.01 * x[:, 1]
    return f


def jeffsfav(x, ns=1):
    global noise
    val = np.sum(x**2)
    f = val + np.random.normal(0, noise / sqrt(ns))
    return f


def jeffsfav_true(x):
    val = np.sum(x**2)
    return val


#' Computes variance reduction at x given a new design xnew (s_n^2(x) - s_n+1^2(x))
#' @param model hetGP model
#' @param x design where to compute the reduction at (default is at \code{xnew})
#' @param xnew new design matrix
#' @param nr number of replicates at each xnew (default 1)
#' @param forceSym force to return a symmetric matrix (default to \code{TRUE})
def vared(model, xnew, x=None, nr=None, forceSym=True):
    if nr is None:
        nr = np.ones(xnew.shape[0])
    if model["trendtype"] != "SK":
        print("This function is intented for simple kriging")

    knxnew = cov_gen(X1=xnew, X2=model["X0"], theta=model["theta"], type=model["covtype"])
    cnxnew = cov_gen(X1=xnew, theta=model["theta"], type=model["covtype"])
    new_lambda = model.predict(x=xnew, nugs_only=True)["nugs"] / model["nu_hat"]
    vn = cnxnew - knxnew @ model["Ki"] @ knxnew.T + np.diag(new_lambda / nr + model["eps"])
    vni = dtrtri(np.linalg.cholesky(vn).T)[0]
    vni = vni @ vni.T
    gn = -model["Ki"] @ knxnew.T @ vni
    if x is None:
        kg = knxnew @ gn
        res = kg @ vn @ kg.T + cnxnew @ kg.T + kg @ cnxnew + cnxnew.T @ vni @ cnxnew
    else:
        knx = cov_gen(X1=x, X2=model["X0"], theta=model["theta"], type=model["covtype"])
        kxxnew = cov_gen(X1=xnew, X2=x, theta=model["theta"], type=model["covtype"])
        kg = knx @ gn
        res = kg @ vn @ kg.T + kxxnew.T @ kg.T + kg @ kxxnew + kxxnew.T @ vni @ kxxnew

    if np.shape(res)[0] > 1 and forceSym:
        res = 1 / 2 * (res + res.T)
    res = res * model["nu_hat"]
    return res


#' Compute the minimal number of replicates to reach a given total relative variance reduction
#' @details Computes the minimal number of replicates a such that (var(Y(n, x)) - var(Y(n + a, x))) / var(Y(n, x)) > threshold
#' @param xnew design where replicates are added
#' @param x optional design where the variance reduction is computed (if \code{NULL}, both replication and prediction are considered happening at \code{xnew})
#' @param model GP model
#' @param threshold relative variance reduction in [0, 1]
#' @param maxrep maximal value for a
#' @param curvar (optional) current variance
#' @param rounding if TRUE (default), returns the smallest integer getting the condition.
#' @return the number of replicates just before the condition is met
#' @note  Potential speed up if integrating the code from vared
def tot_rep_thres(model, xnew, threshold, x=None, curvar=None, rounding=True):
    knxnew = cov_gen(X1=xnew, X2=model["X0"], theta=model["theta"], type=model["covtype"])
    cnxnew = cov_gen(X1=xnew, theta=model["theta"], type=model["covtype"])
    new_lambda = model.predict(x=xnew, nugs_only=True)["nugs"] / model["nu_hat"]
    if x is None:
        if curvar is None:
            sn2xnew = cnxnew - knxnew @ model["Ki"] @ knxnew.T
        else:
            sn2xnew = curvar / model["nu_hat"]
        nr = new_lambda / (sn2xnew / threshold - sn2xnew)
    else:
        knx = cov_gen(X1=x, X2=model["X0"], theta=model["theta"], type=model["covtype"])
        kxxnew = cov_gen(X1=xnew, X2=x, theta=model["theta"], type=model["covtype"])
        sn2xxnew = kxxnew - knxnew @ model["Ki"] @ knx.T
        sn2xnew = cnxnew - knxnew @ model["Ki"] @ knxnew.T
        sn2x = 1 - knx @ model["Ki"] @ knx.T
        if sn2xxnew ^ 2 / (sn2x * sn2xnew) < threshold:
            nr = np.inf
        else:
            nr = new_lambda / (sn2xxnew ^ 2 / (threshold * sn2x) - sn2xnew)
    nr.reshape(1)
    if rounding:
        nr = np.ceil(nr)
    return np.maximum(nr, 1)


#' Deprecated version
#' Compute the minimal number of replicates to reach a given total relative variance reduction
#' @details Computes the minimal number of replicates a such that (var(Y(n, x)) - var(Y(n + a, x))) / var(Y(n, x)) > threshold
#' @param xnew design where replicates are added
#' @param x optional design where the variance reduction is computed (if \code{NULL}, both replication and prediction are considered happening at \code{xnew})
#' @param model GP model
#' @param threshold relative variance reduction in [0, 1]
#' @param maxrep maximal value for a
#' @param curvar (optional) current variance
#' @return the number of replicates just before the condition is met
#' @note  Potential speed up if integrating the code from vared
def tot_rep_thres_v0(model, xnew, threshold, maxrep, x=None, curvar=None):
    if curvar is None:
        if x is None:
            curvar = model.predict(xnew)["sd2"]
        else:
            curvar = model.predict(x)["sd2"]

    curvar = np.maximum(curvar, sqrt(np.finfo(float).eps))
    a = 1
    b = maxrep
    if vared(model=model, x=x, xnew=xnew, nr=a) / curvar >= threshold:
        return a
    if vared(model=model, x=x, xnew=xnew, nr=b) / curvar < threshold:
        return np.inf
    while b - a > 1:
        if vared(model=model, x=x, xnew=xnew, nr=round((a + b) / 2)) / curvar > threshold:
            b = np.round((a + b) / 2)
        else:
            a = np.round((a + b) / 2)

    return b


#' Update existing hetGP list with new observations
#' Seems to work in place by modifying the Xlist object
#' @param Xlist Formatted list to be updated
#' @param xnew new design
#' @param znew new observation
#' @param ik index in Xlist of xnew, if known, WARNING: otherwise xnew is assumed to be a new point
def add_reps(Xlist, xnew, znew, ik=None):
    if Xlist["outputStats"] is not None:
        znew = (znew - Xlist["outputStats"][0]) / sqrt(Xlist["outputStats"][1])
    if ik is None:
        Xlist["X0"] = np.concatenate((Xlist["X0"], xnew))
        Xlist["Z0"] = np.append(Xlist["Z0"], np.mean(znew))
        Xlist["Z"] = np.append(Xlist["Z"], znew)
        Xlist["mult"] = np.append(Xlist["mult"], np.size(znew))
        Xlist["Zlist"].update({len(Xlist["Z0"]) - 1: np.atleast_1d(znew)})
    else:
        Xlist["Z0"][ik] = (Xlist["mult"][ik] * Xlist["Z0"][ik] + np.sum(znew)) / (Xlist["mult"][ik] + np.size(znew))
        Xlist["Zlist"][ik] = np.append(Xlist["Zlist"][ik], znew)
        idZ = np.cumsum(Xlist["mult"])
        Xlist["Z"] = np.insert(arr=Xlist["Z"], values=znew, obj=idZ[ik])
        Xlist["mult"][ik] = Xlist["mult"][ik] + np.size(znew)


# #' Bootstrap X and Z in order to have multiples of ns observations, if necessary
# #' @noRd
# def boot_reps(Xlist, ns):
#     for i in np.arange(Xlist["X0"].shape[0]):
#         if Xlist["mult"][i] > 1:
#             if Xlist["mult"][i] % ns == 0:
#                 Xlist["Zlist"][i] = np.array(np.mean(np.reshape(Xlist["Zlist"][i], [-1, ns]), axis=1))
#                 Xlist["mult"][i] = Xlist["mult"][i] / ns
#             else:
#                 Xnewboot = np.random.choice(Xlist["Zlist"][i], size=int(np.ceil(Xlist["mult"][i] / ns) * ns), replace=True)
#                 Xlist["Zlist"][i] = np.array(np.mean(np.reshape(Xnewboot, newshape=[-1, ns]), axis=1))
#                 Xlist["mult"][i] = np.size(Xnewboot) / ns
#             Xlist["Z0"][i] = np.mean(Xlist["Zlist"][i])
#     # Else: do nothing (cannot have fractionnal mult)
#     Xlist["Z"] = np.concatenate([Xlist["Zlist"].get(k) for k in np.arange(Xlist["X0"].shape[0])], axis=0)


#' Reduce X and Z to multiples of ns observations (remaining groups of less than ns observations are discarded)
#' @noRd
def reduce_reps(Xlist, ns):
    idskeep = np.full(Xlist["mult"].shape[0], False)
    Zlistnew = dict()
    for i in np.arange(Xlist["X0"].shape[0]):
        if Xlist["mult"][i] > 1:
            if Xlist["mult"][i] > ns:
                idskeep[i] = True
                if Xlist["mult"][i] % ns == 0:
                    Xlist["Zlist"][i] = np.array(np.mean(np.reshape(Xlist["Zlist"][i], [-1, ns]), axis=1))
                    Xlist["mult"][i] = Xlist["mult"][i] / ns
                else:
                    Xnewboot = np.random.choice(Xlist["Zlist"][i], size=int((Xlist["mult"][i] // ns) * ns), replace=False)
                    Xlist["Zlist"][i] = np.array(np.mean(np.reshape(Xnewboot, shape=[-1, ns]), axis=1))
                    Xlist["mult"][i] = np.size(Xnewboot) / ns
                Zlistnew.update({len(Zlistnew): Xlist["Zlist"][i]})
                Xlist["Z0"][i] = np.mean(Xlist["Zlist"][i])

    Xlist["X0"] = Xlist["X0"][idskeep]
    Xlist["Z0"] = Xlist["Z0"][idskeep]
    Xlist["mult"] = Xlist["mult"][idskeep]
    Xlist["Zlist"] = Zlistnew
    Xlist["Z"] = np.concatenate([Xlist["Zlist"].get(k) for k in np.arange(Xlist["X0"].shape[0])], axis=0)


#' Local EI optimization
#' @param model GP model
#' @param method how to choose the next design: EI (lhs + optim), EIg (lhs + genoud), c("EI", "EIg", "TurBO", "LineSearch")
#' @param lowerTR,upperTR limits of the TR search space
#' @param ncand number of initial candidates for Turbo/EI
#' @references TurBO
#' @note TR center is at 0
def local_acq_search(model, lowerTR, upperTR, method="EI", ncand=None, xc=None, maxrep=None, threshold=None, c0=None, c1=None):
    d = model["X0"].shape[1]

    # Start predictive mean optimization from TR center
    if method == "LineSearch":

        def fm(x):
            if x.shape.__len__() < 2:
                x = np.atleast_2d(x)
            return model.predict(x)["mean"]

        res = optimize.minimize(
            fun=fm,
            x0=np.zeros(d),
            method="L-BFGS-B",
            bounds=[(l, u) for l, u in zip(lowerTR, upperTR)],
        )

        # Check that it is not too close to the starting point
        if np.sum(res["x"] ** 2) < np.finfo(float).eps:
            res["x"] = np.random.uniform(d) * (upperTR - lowerTR) + lowerTR
            print("Local optimization failed, random selecting a new design instead.")

        return res

    ptmp = model.predict(x=model["X0"])
    xmin = model["X0"][np.argmin(ptmp["mean"]), :]
    cst = np.min(ptmp["mean"])

    if method == "costRI":

        def neg_crit_qRI_rep(xnew, xc, xmin, model, maxrep, c0=0, c1=0, cst=None, preds=None, method="fast", penalty=-100, digits=6):
            return -crit_qRI_rep(xnew=xnew, xc=xc, xmin=xmin, model=model, maxrep=maxrep, c0=c0, c1=c1, cst=cst, preds=preds, method=method, penalty=penalty, digits=digits)

        res_g = optimize.differential_evolution(
            func=neg_crit_qRI_rep,
            args=(xc, xmin, model, maxrep, c0, c1, cst),
            bounds=[(l, u) for l, u in zip(np.concatenate((lowerTR, lowerTR, np.zeros(2))), np.concatenate((upperTR, upperTR, maxrep * np.ones(2))))],
            maxiter=40,
            popsize=15,
        )

        res = optimize.minimize(
            x0=res_g["x"],
            fun=neg_crit_qRI_rep,
            args=(xc, xmin, model, maxrep, c0, c1, cst),
            bounds=[(l, u) for l, u in zip(np.concatenate((lowerTR, lowerTR, np.zeros(2))), np.concatenate((upperTR, upperTR, maxrep * np.ones(2))))],
            method="L-BFGS-B",
        )
        nr = np.maximum(res["x"][[2 * d, 2 * d + 1]], 0)
        xnews = np.reshape(res["x"][np.arange(res["x"].shape[0] - 2)], (2, d))
        xnew = xnews[np.argmax(nr), :]
        nr = np.max(nr)
        return dict(par=xnew, value=-res["fun"], nr=nr)

    if method == "costRI2":

        def neg_crit_qRI_rep2(xnew, xc, xmin, model, maxrep, c0=0, c1=0, cst=None, preds=None, method="fast", penalty=-100, digits=6):
            return -crit_qRI_rep2(xnew=xnew, xc=xc, xmin=xmin, model=model, maxrep=maxrep, c0=c0, c1=c1, cst=cst, preds=preds, method=method, penalty=penalty, digits=digits)

        res_g = optimize.differential_evolution(
            func=neg_crit_qRI_rep2,
            args=(xc, xmin, model, maxrep, c0, c1, cst),
            bounds=[(l, u) for l, u in zip(np.concatenate((lowerTR, lowerTR, np.zeros(2))), np.concatenate((upperTR, upperTR, maxrep * np.ones(2))))],
            maxiter=40,
            popsize=50,
        )

        res = optimize.minimize(
            x0=res_g["x"],
            fun=neg_crit_qRI_rep2,
            args=(xc, xmin, model, maxrep, c0, c1, cst),
            bounds=[(l, u) for l, u in zip(np.concatenate((lowerTR, lowerTR, np.zeros(2))), np.concatenate((upperTR, upperTR, maxrep * np.ones(2))))],
            method="L-BFGS-B",
        )
        nr = np.maximum(res["x"][[2 * d, 2 * d + 1]], 0)
        xnews = np.reshape(res["x"][np.arange(res["x"].shape[0] - 2)], (2, d))
        xnew = xnews[np.argmax(nr), :]
        nr = np.max(nr)
        return dict(par=xnew, value=-res["fun"], nr=nr)

    # First: Turbo style by sampling
    if ncand is None:
        ncand = np.minimum(100 * d, 5000)
    if ncand < 500:
        Xcand = lhspy(d, samples=ncand, criterion="m") * (upperTR - lowerTR) + lowerTR
    else:
        Xcand = lhspy(d, samples=ncand) * (upperTR - lowerTR) + lowerTR

    if method == "TurBO" and d > 20:
        Xcand[(np.random.uniform(size=ncand * d) > 20 / d).nonzero()] = 0

    # cst = model.predict(x=np.atleast_2d(np.zeros(d)))["mean"]
    eis = np.zeros(ncand)
    for i in np.arange(ncand):
        eis[i] = crit_EI(x=Xcand[i, :], model=model, cst=cst)[0]

    par = Xcand[eis.argmax(0), :]
    value = np.max(eis)

    def neg_crit_EI(x, model, cst=None, preds=None):
        return -crit_EI(x=x, model=model, cst=cst, preds=preds)[0]

    if method == "EI":
        res = optimize.minimize(
            x0=par,
            fun=neg_crit_EI,
            args=(model, cst),
            bounds=[(l, u) for l, u in zip(lowerTR, upperTR)],
            method="L-BFGS-B",
        )
        # res = crit_optim(model = model, crit = "crit_EI", h = 0, cst = cst)
        if -res["fun"] > np.max(eis):
            par = res["x"]
            value = -res["fun"]

    nr = None

    if method == "qRIauto" or method == "qRIautocons":
        if method == "qRIauto":
            mtype = "totrep"
        else:
            mtype = "consrep"
        # ptmp = model.predict(x = model["X0"])
        # xmin = model["X0"][np.argmin(ptmp["mean"]),:]
        # cst = np.min(ptmp["mean"])

        def neg_crit_qRIauto(x, xc, xmin, model, maxrep, cst=None, threshold=0.05, type="totrep", preds=None, method="fast", digits=6, returnnr=True):
            return -crit_qRI_auto(xnew=x, xc=xc, xmin=xmin, model=model, maxrep=maxrep, cst=cst, threshold=threshold, type=type, preds=preds, method=method, digits=digits, returnnr=returnnr)

        res = optimize.minimize(
            x0=par,
            fun=neg_crit_qRIauto,
            args=(xc, xmin, model, maxrep, cst, threshold, mtype, None, "fast", 6, False),
            bounds=[(l, u) for l, u in zip(lowerTR, upperTR)],
            method="L-BFGS-B",
        )
        critval = crit_qRI_auto(res["x"], xc=xc, xmin=xmin, maxrep=maxrep, cst=cst, model=model, threshold=threshold, type=mtype)
        nr = critval[1]
        value = critval[0]

    # # Check that it is not too close to the starting point
    if np.sum(par**2) < np.finfo(float).eps:
        par = np.random.uniform(d) * (upperTR - lowerTR) + lowerTR
        print("Local optimization failed, random selecting a new design instead.")

    return dict(par=par, value=value, nr=nr)


#' Variance of Y(x) over TR domain
#' @param model homGP or hetGP model
#' @param unitdomain boolean, if \code{TRUE}, \code{x} is assumed to be in [0,1]^d, else in [-1, 1]^d
def VarTR(model, unitdomain):
    if unitdomain:
        # [0, 1]^d
        W = Wij(model["X0"], theta=model["theta"], type=model["covtype"])  # Covariance of the kernel
        m = mi(model["X0"], theta=model["theta"], type=model["covtype"])  # Integral of the kernel
    else:
        # [-1, 1]^d
        if model["covtype"] == "Gaussian":
            theta = model["theta"] / 4
        else:
            theta = model["theta"] / 2
        W = Wij((model["X0"] + 1) / 2, theta=theta, type=model["covtype"])  # Covariance of the kernel
        m = mi((model["X0"] + 1) / 2, theta=theta, type=model["covtype"])  # Integral of the kernel

    imse = model["nu_hat"] * (1 - np.sum(model["Ki"] * W))
    mnv = (model["Ki"] @ model["Z0"]).T @ (W - np.atleast_2d(m).T @ np.atleast_2d(m)) @ (model["Ki"] @ model["Z0"])
    return dict(tot=imse + mnv, imse=imse, mnv=mnv)


def OGPIT(
    func,
    Low,
    Upp,
    nfmax,
    gammam=0.8,
    delta=0.1,
    mindelta=1e-4,
    maxdelta=0.5,
    mintheta=0.2,
    maxtheta=5,
    trace=0,
    vredthrestot=0.2,
    maxrep=500,
    beta=1e-3,
    eta1=0.2,
    maxnn=None,
    minnn=None,
    relvarxnew=4,
    deter=False,
    modtype="homGP",
    iso="default",
    ncand="small",
    normalize=True,
    toldist=1e-4,
    boots=False,
    ninit=None,
    Xinit=None,
    Zinit=None,
    ikinit=None,
    lightreturn=True,
    acqtype="EI",
    saveall=True,
    c0=None,
    c1=None,
    imsevar=10,
    speed=True,
    bootns=25,
    maxcost=np.inf,
):
    """Trust region with Gaussian processes in the noisy case

    Parameters
    ----------
    func black box function to be optimized
    ninit = [int] size of the initial design (default: twice the number of variables)
    nfmax = [int] maximum number of evaluation
    Low  = [dbl] [1-by-n] vector of lower bounds
    Upp  = [dbl] [1-by-n] vector of lower bounds
    delta = [dbl] initial TR radius
    min_delta,max_delta = [dbl] min and max TR radius
    min_theta,max_theta = [dbl] min and max scalar lengthscale value
    trace = [int] if greater than 0, print some information
    vredthrestot = [dbl] minimum variance reduction when adding a new observation (default value: 20%)
    maxrep = [int]  maximum number of replicates at once for a given design (can be increased if boots is True)
    beta = [dbl] TR parameter
    eta1 = [dbl TR parameter
    gammam = [dbl] trust region contraction rate
    minnn,maxnn = [int] minimum and maximum number of design points in the TR. For the minimum, it could be d+1 or 2d (default to the harmonic mean of these values). For the maximum 200 is a tradeoff between speed and accuracy.
    relvarxnew = [dbl] number of times the variance at the new design can be compared to the variance of the TR center
    deter = [bool] is the problem deterministic or not (i.e., noisy). Mostly use deter for debugging.
    modtype = [str] either 'homGP' or 'hetGP' (the former learns the noise variance but is slower)
    iso = [bool] or [str] if True, only one lenghscale is used for the GP (faster), i.e., isotropic covariance  or "default" meaning isotropic if the dimension is larger than 10
    ncand = [str] either 'large' or 'small', for the budget dedicated to the acquisition function optimization
    normalize = [bool] to normalize function values in the TR.
    toldist = [dbl] minimal distance in the TR between the new point and an existing one
    boots = [bool] if True, increase the number of replication by bootns when the signal to noise ratio of the GP becomes small
    Xinit, Zinit initial designs and corresponding values
    ikinit = [init] if Xinit and Zinit are provided, the initial center index in Xinit can be provided
    acq_type = [str] criterion name to the select the next design
    saveall = [bool]  If True saves all observations (if boots is True, Zall still contains all observations)
    c0,c1 = [dbl] if setup and replication costs are taken into account
    imsevar = [dbl] coefficient in the comparison between variance of the signal and total variance
    speed = [bool] if True, retraining of the GP model is not done every iteration (the schedule depends on the number of designs in the TR)
    bootns = [int] when boot is triggered, multiply number of evaluations by this value
    maxcost = [int] if c0 and c1 are given, can provide a limit total cost as stopping criterion
    """

    # 0 - Initial Design of experiments
    d = Upp.shape[0]  # number of variables
    if gammam is None:
        gammam = 0.8
    if beta is None:
        beta = 1e-3
    if eta1 is None:
        eta1 = 0.2
    if ninit is None:
        ninit = np.maximum(10, 2 * d)
    if minnn is None:
        minnn = np.round(4 * (d + 1) * d / (3 * d + 1))  # Harmonic mean between d+1 and 2xd
    if maxnn is None:
        maxnn = np.maximum(200, 10 * d)
    if mindelta is None:
        mindelta = 1e-4
    if maxdelta is None:
        maxdelta = 0.5
    eps = np.finfo(float).eps
    gdeter = None

    if deter:
        maxrep = 1
        gdeter = sqrt(eps)

    mleFun = hgp.homGP
    # only homGP at this stage
    # if modtype == "hetGP":
    #     mleFun = hgp.hetGP

    if iso == "default":
        if d > 10:
            iso = True
        else:
            iso = False

    if not iso and not mintheta is None and np.size(mintheta) < d:
        mintheta = mintheta * np.ones(d)
    if not iso and not maxtheta is None and np.size(maxtheta) < d:
        maxtheta = maxtheta * np.ones(d)

    if ncand == "small":
        ncand = min(10 * d, 500)
    else:
        ncand = min(100 * d, 5000)

    curcost = 0
    if maxcost is None:
        maxcost = np.inf
    ninit = max(d + 1, ninit)

    # X is in [0, 1]^d (mapped to [lower, upper])
    # so is Xlist$X0
    # Xks is in [-1, 1]^d (corresponding to the TR of radius delta)
    if Xinit is None:
        # X = np.array([[0.3210, 0.8220], [0.6930, 0.1390], [0.0926, 0.4940], [0.4730, 0.5960], [0.7330, 0.6740], [0.5520, 0.3730], [0.8740, 0.2540], [0.2620, 0.7000], [0.1630, 0.0609], [0.9220, 0.9840]])
        X = lhspy(d, samples=ninit, criterion="m")  # Maximum Latin Hypercube design to start with.
        Z = np.zeros(ninit)
        for i in range(0, ninit):
            Z[i] = func(X[i, :] * (Upp - Low) + Low)
        n = int(ninit)
        if c0 is not None:
            curcost = ninit * (c0 + c1)
    else:
        if Xinit.shape[0] != np.size(Zinit):
            print("The number of rows of Xinit does not match the length of Zinit.")
        X = Xinit
        Z = Zinit
        n = int(0)
        if Xinit.shape[0] < ninit:
            Xcomp = lhspy(d, samples=ninit - Xinit.shape[0])
            Zcomp = np.zeros(ninit - Xinit.shape[0])
            for i in range(0, ninit - Xinit.shape[0]):
                Zcomp[i] = func(Xcomp[i, :] * (Upp - Low) + Low)
            X = np.vstack((X, Xcomp))
            Z = np.concatenate((Z, Zcomp))
            n = Xcomp.shape[0]
        if c0 is not None:
            curcost = (ninit - Xinit.shape[0]) * (c0 + c1)

    Xall = np.atleast_2d(X)
    Zall = None  # used to save the observations if ns becomes larger than 1

    # Use hetGP replicate structure (i.e., X0 matrix of unique designs, Z0 matrix of means over replicates,
    # mult the number of replicates per unique design
    Xlist = find_reps(X, Z)

    if ikinit is None:
        ik = np.argmin(Xlist["Z0"])  # index of the TR center in X and Z
    else:
        ik = ikinit
    xk = Xlist["X0"][ik, :]  # TR center
    zk = Xlist["Z0"][ik]  # TR center value
    outxks = xk  # Store TR center values
    outxnews = np.empty(shape=(0, d))  # Store new design center values
    evalits = np.zeros(1)  # Store the number of evaluations at each iteration
    xc = np.zeros(d)  # Center of the TR in the coordinates used by the GP (i.e., in [-1, 1]^d)
    suc = True  # Was the previous step successful?
    ns = 1
    increasens = False
    settings = dict(trace=trace)

    while n < nfmax and delta > mindelta and curcost < maxcost:

        # Increase the amount of unitary runs of the function for one evaluation
        if not deter and boots and n > ninit and nin >= np.minimum(4 * (d + 1), maxnn) and (increasens or (nnewrep == maxrep and (model["g"] > 1e2 - 1 or nin == maxnn))):
            if model["g"] > 1e2 - 1 and trace > 0:
                print("Increase ns due to g value > 99.")
            if nin == maxnn and trace > 0:
                print("Increase ns due to the number of designs in the TR.")
            if increasens and trace > 0:
                print("Increase ns due to increasens.")

            reduce_reps(Xlist=Xlist, ns=bootns)
            ik = np.flatnonzero((xk == Xlist["X0"]).all(1))
            if np.size(ik) == 0:
                ik = Xlist["X0"].shape[0]
                xk = Xlist["X0"][ik, :]

            if ns == 1:
                Zall = Z
            ns = ns * bootns
            increasens = False
            suc = True
            maxdelta = delta  # prevent re-growing the TR

        # Find nns nearest neighbors to the center
        ndists = np.max(np.abs(Xlist["X0"] - xk), axis=1)
        nntmp = min(maxnn, Xlist["X0"].shape[0], max(np.sum(ndists <= delta + np.minimum(np.maximum(eps, delta / 1e4), np.sqrt(eps))), 2))  # Take all points in the TR, and some outside if needed
        nns = np.argsort(ndists)[np.arange(nntmp)]

        # Extract NN points and rescale (around the TR center)
        Xks = Xlist["X0"][nns]

        # xk should be in first position in Xks
        if np.sum(xk) < np.sum(Xks[0, :]) or np.sum(xk) > np.sum(Xks[0, :]):
            print("xk is not in first position in Xks")

        # Rescale such that the TR correspond to [-1,1]^d
        Xks = (Xks - xk) / delta
        lowerTR = (np.maximum(xk - delta, Low) - xk) / delta
        upperTR = (np.minimum(xk + delta, Upp) - xk) / delta

        # Number of points in the TR
        iin = np.array(np.where(np.max(np.abs(Xks), axis=1) <= 1 + np.minimum(np.maximum(eps, delta / 1e4), np.sqrt(eps)))).flatten()
        nin = len(iin)
        if trace > 0:
            print("Number of designs in the TR:", nin)

        # Fit GP model
        beta0 = np.mean(Xlist["Z0"][nns[iin]])
        if normalize and nin > 1:
            osts = [beta0, np.std(Xlist["Z0"][nns])]
            beta0 = 0
        else:
            osts = [0, 1]

        minthetatmp = mintheta
        maxthetatmp = maxtheta
        if minthetatmp is None or maxthetatmp is None:
            try:
                autothetas = auto_bounds(X=Xks, covtype="Matern5_2", min_cor=0.2, max_cor=0.9)
            except:
                autothetas = dict(lower=0.2 * np.ones(d), upper=5 * sqrt(d) * np.ones(d))
            if minthetatmp is None:
                if iso:
                    minthetatmp = np.mean(autothetas["lower"])
                else:
                    minthetatmp = autothetas["lower"]
            if maxthetatmp is None:
                if iso:
                    maxthetatmp = np.mean(autothetas["upper"])
                else:
                    maxthetatmp = autothetas["upper"]

        # Avoid relearning the hyperparameters at every iteration when there is already sufficient data
        if speed and not suc and nin > 5 * d and ((nin < 50 and nin % 3 != 0) or (nin < 100 and nin % 5 != 0) or (nin >= 100 and nin % 10 != 0)):
            knownparams = dict(theta=model["theta"], beta0=beta0, g=model["g"])
        else:
            knownparams = dict(beta0=beta0, g=gdeter)
            if suc:
                init = dict()
                settings = dict(trace=trace)
            else:
                init = dict(theta=model["theta"], g=model["g"])
                settings = dict(trace=trace, factr=1e8)

        noiseControl = dict(g_bounds=[sqrt(eps), np.maximum(100, np.minimum(1e5, np.var(np.concatenate([Xlist["Zlist"].get(k) for k in nns], axis=0)) / np.var(Xlist["Z0"][nns])))])

        model = mleFun()
        model.mle(
            dict(
                X0=Xks,
                mult=Xlist["mult"][nns],
                Z0=(Xlist["Z0"][nns] - osts[0]) / osts[1],
            ),
            Z=(np.concatenate([Xlist["Zlist"].get(k) for k in nns], axis=0) - osts[0]) / osts[1],
            covtype="Matern5_2",
            settings=settings,
            lower=minthetatmp,
            upper=maxthetatmp,
            noiseControl=noiseControl,
            init=init,
            known=knownparams,
        )

        if nntmp > 10 * d:
            if iso:
                knowntheta = model["theta"][0]
            else:
                knowntheta = model["theta"].flatten()
            knowng = model["g"]
        else:
            knowntheta = None
            knowng = None

        if deter:
            knowng = gdeter

        if trace > 0:
            print("Var", np.var(np.concatenate([Xlist["Zlist"].get(k) for k in nns], axis=0)) / np.var(Xlist["Z0"][nns]), "nu", model["nu_hat"], "g: ", model["g"], " ns:", ns)

        # Check that there are sufficient design points to build a GP (and replicate them sufficiently)
        if nin < minnn and n < nfmax - minnn - 1:
            npois = minnn - nin
            Xpois = np.reshape(np.random.uniform(lowerTR.repeat(npois), upperTR.repeat(npois), npois * d), [d, npois]).T
            nreppois = np.ones(npois, dtype="int")
            if nin > 2:  # Model is not relevant with too few designs
                for i in range(npois):
                    nreppois[i] = np.minimum(maxrep, tot_rep_thres(xnew=np.atleast_2d(Xpois[i, :]), model=model, threshold=vredthrestot)[0, 0])

            if n + sum(nreppois) * ns - ns > nfmax:
                nreppois = np.ones(npois, dtype="int")
            npois = np.sum(nreppois)
            Xpois = Xpois * delta + xk

            for i in range(Xpois.shape[0]):
                zpois = np.zeros(nreppois[i])
                for j in range(nreppois[i]):
                    if saveall and ns > 1:
                        zpoistmp = np.zeros(ns)
                        for kk in np.arange(ns):
                            zpoistmp[kk] = func(Xpois[i, :] * (Upp - Low) + Low, ns=1)
                        Zall = np.concatenate((Zall, zpoistmp))
                        zpois[j] = np.mean(zpoistmp)
                    else:
                        zpois[j] = func(Xpois[i, :] * (Upp - Low) + Low, ns=ns)

                Z = np.concatenate((Z, zpois))
                add_reps(Xlist=Xlist, xnew=np.atleast_2d(Xpois[i, :]), znew=zpois)

            X = np.vstack((X, Xpois.repeat(nreppois, axis=0)))
            Xall = np.vstack((Xall, Xpois.repeat(nreppois, axis=0)))
            n = n + npois * ns
            if trace > 0:
                print("Added", nreppois * ns, "designs for poisedness")
            if c0 is not None:
                curcost = curcost + c0 * npois + c1 * np.sum(nreppois) * ns
            continue

        # Next point selection
        afopt = local_acq_search(model=model, lowerTR=lowerTR, upperTR=upperTR, xc=xc, ncand=ncand, method=acqtype, maxrep=maxrep, threshold=vredthrestot, c0=c0, c1=c1)
        if acqtype in ("qRIauto", "costRI") and trace > 0:
            print("nr:", afopt["nr"])

        if not deter and toldist > 0:
            # Check if new design is not to close to existing design
            dists = np.sqrt(euclidean_dist(np.atleast_2d(afopt["par"]), model["X0"]))

            if np.min(dists) < toldist:
                afopt["par"] = model["X0"][np.argmin(dists), :]

        sk = afopt["par"]

        # Prepare rho test: compute denominator
        ploo_sk = model.predict(x=np.atleast_2d(sk))
        ploo_xk = LOO_preds(model=model, ids=1)

        # Replication at xnew
        # Future variance should not be relvarxnew times larger than the one at the center.
        vrednew = np.maximum(vredthrestot, (ploo_sk["sd2"] - model.predict(x=np.atleast_2d(xc))["sd2"] * relvarxnew) / np.maximum(ploo_sk["sd2"], eps))

        if afopt["nr"] is None:
            nnewrep = tot_rep_thres(xnew=np.atleast_2d(sk), model=model, threshold=vrednew[0], curvar=ploo_sk["sd2"])
        else:
            nnewrep = afopt["nr"]
            nnewrep = np.maximum(nnewrep, tot_rep_thres(xnew=np.atleast_2d(sk), model=model, threshold=vrednew[0], curvar=ploo_sk["sd2"]))

        nnewrep = int(np.maximum(1, np.min(np.array([nnewrep[0] * ns, maxrep * ns, nfmax - int(n)], dtype="object")) / ns))

        # Evaluate new point
        xnew = xk + sk * delta
        outxnews = np.vstack((outxnews, xnew))

        znew = np.zeros(nnewrep)

        for i in range(nnewrep):
            if saveall and ns > 1:
                znewtmp = np.zeros(ns)
                for j in range(ns):
                    znewtmp[j] = func(xnew * (Upp - Low) + Low, ns=1)
                Zall = np.concatenate((Zall, znewtmp))
                znew[i] = np.mean(znewtmp)
            else:
                znew[i] = func(xnew * (Upp - Low) + Low, ns=ns)

        Xall = np.vstack((Xall, np.atleast_2d(xnew).repeat(nnewrep * ns, axis=0)))

        X = np.vstack((X, np.atleast_2d(xnew).repeat(nnewrep, axis=0)))
        Z = np.concatenate((Z, znew))
        n = n + nnewrep * ns

        if trace > 0:
            print("New point replicated ", nnewrep * ns, "times.")
        if c0 is not None:
            curcost = curcost + c0 + c1 * nnewrep * ns

        inew = np.flatnonzero((xnew == Xlist["X0"]).all(1))  # supposedly faster than np.argwhere((v == A).all(1))

        if inew.shape[0] > 0:
            # xnew is a replicate
            inew = inew[0]
            add_reps(Xlist=Xlist, xnew=np.atleast_2d(xnew), znew=znew, ik=inew)
        else:
            # xnew is a new design
            inew = Xlist["X0"].shape[0]
            add_reps(Xlist=Xlist, xnew=np.atleast_2d(xnew), znew=znew)

        # Update of the GP model
        idtmp = np.unique(np.concatenate((nns, np.array([inew]))))
        beta0 = np.mean(Xlist["Z0"][np.unique(np.concatenate((nns[iin], np.array([inew]))))])
        if normalize:
            osts = [beta0, np.std(Xlist["Z0"][idtmp])]
            beta0 = 0
        else:
            osts = [0, 1]

        model = mleFun()
        model.mle(
            dict(
                X0=(Xlist["X0"][idtmp] - xk) / delta,
                mult=Xlist["mult"][idtmp],
                Z0=(Xlist["Z0"][idtmp] - osts[0]) / osts[1],
            ),
            Z=(np.concatenate([Xlist["Zlist"].get(k) for k in idtmp], axis=0) - osts[0]) / osts[1],
            covtype="Matern5_2",
            settings=dict(trace=trace),
            lower=minthetatmp,
            upper=maxthetatmp,
            noiseControl=noiseControl,
            known=dict(beta0=beta0, theta=knowntheta, g=knowng),
        )

        # Prediction at TR center and new point
        psk = model.predict(x=np.vstack((xc, sk)), xprime=np.vstack((xc, sk)))  # prediction at the tentative new point
        psk["sd2"] = np.maximum(psk["sd2"], eps)

        # Update TR radius
        suc = False
        if psk["mean"][0] - psk["mean"][1] > beta * min(delta, delta**2):
            # Compute rhok
            suc = True
            if psk["mean"][0] - psk["mean"][1] > 0 and ploo_xk["mean"] - ploo_sk["mean"] < 0:
                rho = (ploo_sk["mean"] - ploo_xk["mean"] + psk["mean"][0] - psk["mean"][1]) / (ploo_sk["mean"] - ploo_xk["mean"])
            else:
                rho = (psk["mean"][0] - psk["mean"][1]) / (ploo_xk["mean"] - ploo_sk["mean"])

            if rho >= eta1:
                delta = min(maxdelta, delta / gammam)
            else:
                VTR = VarTR(model, unitdomain=False)
                if not deter and model["g"] > 1 and nin < maxnn and VTR["mnv"] < imsevar * VTR["imse"]:
                    if trace > 0:
                        print("Variance of predictive mean smaller than IMPSE: cannot decrease the TR (", VTR["mnv"], "vs", VTR["imse"], "). (imsevar = ", imsevar, ")")
                    else:
                        if trace > 0:
                            print("Improvement is too small: rho=", rho, "vs. eta1 =", eta1)
                        delta = delta * gammam

            # Update TR center
            xk = Xlist["X0"][inew]
            ik = inew
            zk = Xlist["Z0"][ik]
            outxks = np.vstack((outxks, xk))

        else:
            VTR = VarTR(model, unitdomain=False)
            if not deter and model["g"] > 1 and nin < maxnn and VTR["mnv"] < imsevar * VTR["imse"]:
                if trace > 0:
                    print("Variance of predictive mean smaller than IMPSE: cannot decrease the TR (", VTR["mnv"], "vs", VTR["imse"], "). (imse var = ", imsevar, ")")
            else:
                delta = delta * gammam

            zk = Xlist["Z0"][ik]
            outxks = np.vstack((outxks, xk))

        evalits = np.concatenate((evalits, n * np.ones(1)))
        if trace > 0:
            print("Iteration:", Xlist["X0"].shape[0], "nevals:", n, ", current center:", xk * (Upp - Low) + Low, "TR radius", delta, ", value:", zk)
    if lightreturn:
        return dict(par=xk * (Upp - Low) + Low, value=zk, X=X * (Upp - Low) + Low, Z=Z)
    else:
        return dict(
            par=xk * (Upp - Low) + Low,
            value=zk,
            X=X * (Upp - Low) + Low,
            Z=Z,
            Xlist=Xlist,
            evalits=evalits,
            nevals=n,
            Xks=outxks * (Upp - Low) + Low,
            Xnews=outxnews * (Upp - Low) + Low,
            delta=delta,
            ipar=ik,
            Xall=Xall * (Upp - Low) + Low,
            Zall=Zall,
        )


#' Function going smoothly from 0 to 1 on [xmin, xmax], with zero gradient at 0 and xlim.
#' @param x value considered in [0, 1]
#' @param xmin value at which the function must be 0
#' @param xmax value at which the function must be 1
def penf(x, xmin=0, xmax=1):
    if x < 0:
        print("x should not be negative")
    if x > xmax:
        return 1
    if x < xmin:
        return 0
    x = (x - xmin) / (xmax - xmin)
    # erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1
    return (scipy.special.erf(np.tan(x * np.pi - np.pi / 2)) + 1) / 2


#' Expected improvement criterion with penalty if too close to TR solution
#' Global EI optimization
#' @param model GP model
#' @param xstarsloc matrix containing the local optima found by TR searches
#' @param ldist threshold distance to existing local optima
def crit_EIp(x, model, ldist, cst=None, preds=None, xstarsloc=None):
    if cst is None:
        cst = np.min(model.predict(model, x=model["X0"])["mean"])
    if len(x.shape) == 1:
        x = x.reshape(-1, model.X0.shape[1])
    if preds is None:
        preds = model.predict(x=x)

    res = crit_EI(x, model=model, cst=cst, preds=preds)

    if xstarsloc is not None:
        dists = np.sqrt(euclidean_dist(x, xstarsloc))
        if np.min(dists) < ldist * 1.1:
            res = res * penf(np.min(dists), xmin=ldist, xmax=ldist * 1.1)

    return res


#' Logarithm of Expected improvement criterion with penalty if too close to TR solution
#' Global EI optimization
#' @param model GP model
#' @param xstarsloc matrix containing the local optima found by TR searches
#' @param ldist threshold distance to existing local optima
def crit_logEIp(x, model, ldist, cst=None, preds=None, xstarsloc=None):
    if cst is None:
        cst = np.min(model.predict(model, x=model["X0"])["mean"])
    if len(x.shape) == 1:
        x = x.reshape(-1, model.X0.shape[1])
    if preds is None:
        preds = model.predict(x=x)

    res = crit_logEI(x, model=model, cst=cst, preds=preds)

    if xstarsloc is not None:
        dists = np.sqrt(euclidean_dist(x, xstarsloc))
        if np.min(dists) < ldist * 1.1:
            res = res + np.maximum(-800, np.log(penf(np.min(dists), xmin=ldist, xmax=ldist * 1.1)))

    return res


#' Global EI optimization
#' @param model GP model
#' @param method how to choose the next design: EI (lhs + optim), EIp (lhs + genoud on penalized EI),
#' @param Low,Upp limits of the search domain
#' @param ncand number of initial candidates for Turbo/EI
#' @param xstarsloc matrix containing the local optima found by TR searches
#' @param ldist minimal distance to xstarsloc to start penalizing
#' @references TurBO
#' @note TR center is at 0
def global_acq_search(model, Low, Upp, ldist, method="EI", ncand=None, xstarsloc=None):
    d = model["X0"].shape[1]

    # First: Turbo style by sampling
    if ncand is None:
        ncand = np.minimum(100 * d, 5000)
    if ncand < 500:
        Xcand = lhspy(d, samples=ncand, criterion="m") * (Upp - Low) + Low
    else:
        Xcand = lhspy(d, samples=ncand, criterion="m") * (Upp - Low) + Low

    Xcand = np.maximum(np.minimum(Xcand, Upp), Low)

    if method == "TurBO" and d > 20:
        Xcand[(np.random.uniform(size=ncand * d) > 20 / d).nonzero()] = 0
        # Xcand[which(runif(ncand * d) > 20/d)] = 0

    cst = np.min(model.predict(x=model["X0"])["mean"])

    def neg_crit_EI(x, model, cst=None, preds=None):
        return -crit_EI(x=x, model=model, cst=cst, preds=preds)[0]

    if method == "EI":
        eis = np.zeros(ncand)
        for i in np.arange(ncand):
            eis[i] = crit_EI(x=Xcand[i, :], model=model, cst=cst)[0]

        par = Xcand[eis.argmax(0), :]
        value = np.max(eis)
        preds = None
        res = optimize.minimize(
            x0=par,
            fun=neg_crit_EI,
            args=(model, cst, preds),
            bounds=[(l, u) for l, u in zip(Low, Upp)],
            method="L-BFGS-B",
        )

        if -res["fun"] > np.max(eis):
            par = res["x"]
            value = -res["fun"]

    def neg_crit_EIp(x, model, ldist, cst=None, preds=None, xstarsloc=None):
        return -crit_EIp(x=x, model=model, ldist=ldist, cst=cst, preds=preds, xstarsloc=xstarsloc)[0]

    if method == "EIp":
        eips = np.zeros(ncand)
        for i in np.arange(ncand):
            eips[i] = crit_EIp(x=Xcand[i, :], model=model, cst=cst, xstarsloc=xstarsloc, ldist=ldist)[0]

        par = Xcand[eips.argmax(0), :]
        value = np.max(eips)

        preds = None
        res = optimize.minimize(
            x0=par,
            fun=neg_crit_EIp,
            args=(model, ldist, cst, preds, xstarsloc),
            bounds=[(l, u) for l, u in zip(Low, Upp)],
            method="L-BFGS-B",
        )

        if -res["fun"] > value:
            par = res["x"]
            value = -res["fun"]

    def neg_crit_logEIp(x, model, ldist, cst=None, preds=None, xstarsloc=None):
        return -crit_logEIp(x=x, model=model, ldist=ldist, cst=cst, preds=preds, xstarsloc=xstarsloc)[0]

    if method == "logEIp":
        eips = np.zeros(ncand)
        for i in np.arange(ncand):
            eips[i] = crit_logEIp(x=Xcand[i, :], model=model, cst=cst, xstarsloc=xstarsloc, ldist=ldist)[0]

        par = Xcand[eips.argmax(0), :]
        value = np.max(eips)

        preds = None
        res = optimize.minimize(
            x0=par,
            fun=neg_crit_logEIp,
            args=(model, ldist, cst, preds, xstarsloc),
            bounds=[(l, u) for l, u in zip(Low, Upp)],
            method="L-BFGS-B",
        )

        if -res["fun"] > value:
            par = res["x"]
            value = -res["fun"]

    return dict(par=par, value=value)


#' New infill criterion
#' @param x where to compute the reduction in improvement at
#' @param xnew where the new designs will be added
#' @param cst current minimum
#' @param nr number of replicates, vector of size nrow(xnew). Default to one replication.
#' @export
def crit_qRIt(x, xnew, model, cst=None, nr=None, preds=None, method="fast", fastCompute=True):
    if cst is None:
        cst = np.min(model.predict(x=model["X0"])["mean"])
    if len(x.shape) == 1:
        x = x.reshape(-1, model.X0.shape[1])
    if len(xnew.shape) == 1:
        xnew = xnew.reshape(-1, model.X0.shape[1])
    if preds is None or preds["cov"] is None:
        preds = model.predict(x=x, xprime=x)
    if nr is None:
        nr = np.ones(xnew.shape[0])

    if np.size(nr) != xnew.shape[0]:
        print("Number of replicates must match the number of new points.")

    if duplicated(x).any():
        x = np.unique(x, axis=0)
        preds = model.predict(x=x, xprime=x)

    if duplicated(xnew).any():
        if xnew.shape[0] == 2:
            xnew = xnew[0, :]
            nr = sum(nr)
        else:
            tmp = find_reps(xnew.repeat(np.arange(xnew.shape[0]), nr), np.repeat(np.nan, np.sum(nr)))
            xnew = tmp["X0"]
            nr = tmp["mult"]

    vred = vared(model, xnew=xnew, x=x, nr=nr)
    if np.min(np.diag(vred)) < 0:
        np.fill_diagonal(vred, np.maximum(np.diag(vred), 0))
    preds["cov"] = 1 / 2 * (preds["cov"] + preds["cov"].T)
    covup = preds["cov"] - vred
    sd2up = np.diag(covup)
    if np.min(np.linalg.eigvals(covup)) <= 1e4 * np.finfo(float).eps:
        sd2up = np.maximum(sd2up, 1e4 * np.finfo(float).eps)
        covup = np.diag(sd2up)

    if method == "fast":
        if x.shape[0] == 1:
            predsup = dict(mean=preds["mean"], sd2=sd2up)
            ei = crit_EI(x, model=model, cst=cst, preds=preds)
            ci = crit_EI(x, model=model, cst=cst, preds=predsup)
        else:
            # cov2cor not available in python, so compute it ourselves
            cov = preds["cov"]
            Dinv = np.diag(1.0 / np.sqrt(np.diag(cov)))  # inverse of diagonal matrix
            cormat = Dinv @ cov @ Dinv
            ei = qEI_cpp(mu=-preds["mean"], s=np.sqrt(preds["sd2"]), cor=cormat, threshold=-cst)
            Dinv = np.diag(1.0 / np.sqrt(np.diag(covup)))  # inverse of diagonal matrix
            corup = Dinv @ covup @ Dinv
            ci = qEI_cpp(mu=-preds["mean"], s=np.sqrt(sd2up), cor=corup, threshold=-cst)

    if np.isnan(ei - ci):
        print("nan")

    return np.maximum(0, ei - ci)


#' Allows to allocate reps without integer constraints
#' @param maxrep maximum number of replicates to be allocated
#' @param c0,c1 fixed costs for evaluating a new evaluation point and for each replicate
#' @param xnew matrix of two new points whose last column is the number of replicates. A vector can be provided, concatenating new points then corresponding number of reps.
#' @param xc,xmin TR center and minimum point
#' @param penalty value used to penalize replication budgets larger than maxrep and lower than 1
#' @param digits how many digits to keep for rounding, default to 6
#' @export
def crit_qRI_rep(xnew, xc, xmin, model, maxrep, c0=0, c1=0, cst=None, preds=None, method="fast", penalty=-100, digits=6):
    d = model["X0"].shape[1]
    pen = 0
    if len(xnew.shape) == 1:
        nr = np.maximum(xnew[[2 * d, 2 * d + 1]], 0)
        xnew = np.reshape(xnew[np.arange(xnew.shape[0] - 2)], (2, d))
    else:
        nr = np.max(0, xnew[:, d + 1])

    if np.max(nr) < 1:
        pen = penalty * (1 - np.max(nr))

    ids = np.where(nr == 0)
    if np.size(ids) > 0:
        xnew = xnew[np.where(nr > 0)[0], :]
        nr = nr[np.where(nr > 0)[0]]

    if np.size(nr) == 0:
        return pen

    if np.sum(nr) > maxrep:
        pen = penalty * (np.sum(nr) - maxrep)

    if np.max(nr) >= 1:
        crit = crit_qRIt(x=np.round(np.vstack((xnew, xc, xmin)), decimals=digits), xnew=np.round(xnew, decimals=digits), cst=cst, model=model, nr=nr, preds=preds, method=method)
    else:
        return pen

    # if(is(crit, "try-error")){
    #   print(xnew)
    #   print(nr)
    #   kkk <- crit_qRIt(x = round(rbind(xnew, xc, xmin), digits = digits), xnew = round(xnew, digits = digits), cst = cst, model = model, nr = nr, preds = preds, method = method)
    #   stop()
    # }
    if c0 + c1 > 0:
        costs = c0 * sum(nr > 0) + c1 * (sum(nr))
    else:
        costs = 1
    return crit / costs + pen


#' Allows to allocate reps without integer constraints - force min(nr) > 1
#' @param maxrep maximum number of replicates to be allocated
#' @param c0,c1 fixed costs for evaluating a new evaluation point and for each replicate
#' @param xnew matrix of new points whose last column is the number of replicates. A vector can be provided, concatenating new points then corresponding number of reps.
#' @param xc,xmin TR center and minimum point
#' @param penalty value used to penalize replication budgets larger than maxrep and lower than 1
#' @param digits how many digits to keep for rounding, default to 6
def crit_qRI_rep2(xnew, xc, xmin, model, maxrep, c0=0, c1=0, cst=None, preds=None, method="fast", penalty=-100, digits=6):
    d = model["X0"].shape[1]
    pen = 0
    if len(xnew.shape) == 1:
        nr = np.maximum(xnew[[2 * d, 2 * d + 1]], 0)
        xnew = np.reshape(xnew[np.arange(xnew.shape[0] - 2)], (2, d))
    else:
        nr = np.max(0, xnew[:, d + 1])

    if np.max(nr) < 1:
        pen = penalty * (1 - np.max(nr))
    if np.min(nr) < 1:
        pen = penalty * (1 - np.min(nr))

    ids = np.where(nr == 0)
    if np.size(ids) > 0:
        xnew = xnew[np.where(nr > 0)[0], :]
        nr = nr[np.where(nr > 0)[0]]

    if np.size(nr) == 0:
        return pen

    if np.sum(nr) > maxrep:
        pen = penalty * (np.sum(nr) - maxrep)

    if np.max(nr) >= 1:
        crit = crit_qRIt(x=np.round(np.vstack((xnew, xc, xmin)), decimals=digits), xnew=np.round(xnew, decimals=digits), cst=cst, model=model, nr=nr, preds=preds, method=method)
    else:
        return pen

    # if(is(crit, "try-error")){
    #   print(xnew)
    #   print(nr)
    #   kkk <- crit_qRIt(x = round(rbind(xnew, xc, xmin), digits = digits), xnew = round(xnew, digits = digits), cst = cst, model = model, nr = nr, preds = preds, method = method)
    #   stop()
    # }
    if c0 + c1 > 0:
        costs = c0 * sum(nr > 0) + c1 * (sum(nr))
    else:
        costs = 1
    return crit / costs + pen


#' @title qRI for one new point (several targets) with auto number of reps
#' This version automatically selects the number of new replicates: compared to the maximum qRI value, it stops when adding one new replicate does not bring more than threshold times maxqRI
#' @param xc,xmin where to compute the reduction in improvement at (including xnew)
#' @param xnew where one new design will be added (only one new design at this stage)
#' @param cst current minimum
#' @param maxrep max number of replicates
#' @param threshold minimal relative qRI reduction brought by one additional new rep
#' @param returnnr is the number of replicates to be returned?
#' @param type either totrep for totrepthreshold or consrep for consrepthreshold
#' @importFrom stats cov2cor
#' @export
def crit_qRI_auto(xnew, xc, xmin, model, maxrep, cst=None, threshold=0.05, type="totrep", preds=None, method="fast", digits=6, returnnr=True):
    if cst is None:
        cst = np.min(model.predict(x=model["X0"])["mean"])
    if len(xnew.shape) == 1:
        xnew = xnew.reshape(-1, model.X0.shape[1])

    if type == "consrep":
        nr = min(maxrep, max(1, cons_rep_thres(xnew=xnew, model=model, threshold=0.01, maxrep=maxrep)))
    else:
        nr = min(maxrep, max(1, tot_rep_thres(xnew=xnew, model=model, threshold=threshold)))

    val = crit_qRIt(x=np.round(np.vstack((xnew, xc, xmin)), decimals=digits), xnew=np.round(xnew, decimals=digits), cst=cst, model=model, nr=nr, preds=preds, method=method)
    if returnnr:
        return np.array((val, nr))
    else:
        return val


#' Global optimization based on EGO and OGPIT
#' @param func function to be minimized
#' @param ... additional arguments of func
#' @param Low,Upp bounds
#' @param nfmax total number of evaluations
#' @param localdelta TR size for local searches
#' @param mindelta minimal size of the TR
#' @param ninit number of initial points
#' @param maxrep maximum number of replicates at once for a given design (can be bypassed if the TR radius is small)
#' @param ldist distance under which a new local search cannot be initiated
#' @param tol_dist for noisy problems, distance under which a replicate is preferred during the global search
#' @param trace if > 0 display information
#' @param afitmax,stpcmax number of times acquisition function can add points close to an existing one before switching to local search, resp. be too small before stopping
#' @param checkinterp should the test on interpolating vs noise error be used to trigger local search?
#' @param maxXu maximum number of unique designs of the global GP
#' @param localcrit,globalcrit which acquisition function to use for the local/global searches
def STOGPIT(
    func,
    Low,
    Upp,
    nfmax,
    maxnn,
    localbudget=None,
    localdelta=1e-2,
    ninit=None,
    maxrep=100,
    ldist=0.1,
    beta0=0,
    globalcrit="logEI",
    localcrit=None,
    deter=False,
    tol_dist=1e-2,
    trace=1,
    boots=True,
    modtype="homGP",
    lightreturn=True,
    ncand=None,
    afitmax=10,
    stpcmax=10,
    mindelta=None,
    maxdelta=None,
    checkinterp=True,
    maxXu=250,
):

    if trace > 1:
        traceogpit = 1
    else:
        traceogpit = 0
    d = np.size(Low)

    if np.array_equal(Low, np.zeros(d)) and np.array_equal(Upp, np.ones(d)):
        Low_o = Low
        Upp_o = Upp
    else:
        Low_o = Low
        Upp_o = Upp

        def func_sc(x, ns=1):
            return func(x * (Upp_o - Low_o) + Low_o, ns=ns)

        Low = np.zeros(d)
        Upp = np.ones(d)

    mleFun = hgp.homGP
    if modtype == "hetGP":
        mleFun = hgp.hetGP

    if beta0 is None:
        beta0 = 0

    if globalcrit == "logEI":
        globalcrit = "logEIp"
    if globalcrit == "EI":
        globalcrit = "EIp"

    if ninit is None:
        ninit = 5 * d
    ninit = np.maximum(5 * d, ninit)
    X = lhspy(d, samples=ninit, criterion="m")
    Z = np.zeros(ninit)
    for i in range(0, ninit):
        Z[i] = func_sc(X[i, :] * (Upp - Low) + Low)
    n = ninit

    eps = 2.3e-16  # 1.5e-8
    gdeter = None

    if deter:
        maxrep = 1
        gdeter = sqrt(eps)
    if maxdelta is None:
        maxdelta = localdelta
    if localbudget is None:
        localbudget = np.floor(nfmax / 3)

    Xall = X  # Store all designs
    Xbest = np.atleast_2d(X[np.argmin(Z), :])  # Store current best design
    nbest = np.atleast_1d(np.argmin(Z))  # Store index when the best changed
    Zbest = np.atleast_1d(np.min(Z))  # Store current best estimation of the minimum value
    lstep = np.repeat(False, ninit)  # Store if local or global step
    bvec = np.repeat(False, ninit)  # for tracking
    localres = dict()  # for tracking

    xk = None  # TR center

    globalit = True  # Switch between local and global optimization

    xstarsloc = None  # Store TR results designs
    fstarsloc = None  # Store TR results mean values
    xstarsloc_all = dict()  # Store TR results designs
    fstarsloc_all = None  # Store all TR results values
    afits = 0  # Count number of unsuccessful acquisition function searches
    stpc = 0  # Count number of times the best acquisition function value is almost zero
    nu = ninit  # Number of unique designs
    mtmp = None

    while n < nfmax and afits <= afitmax and stpc <= stpcmax and nu < maxXu:
        if globalit:
            if mtmp is None or mtmp["Z"].shape[0] != Z.shape[0]:
                model = mleFun()
                model.mle(X=X, Z=Z, covtype="Matern5_2", settings=dict(trace=trace), known=dict(beta0=beta0, g=gdeter))
            else:
                model = mtmp

            nu = model["X0"].shape[0]

            # Run EI search
            afopt = global_acq_search(model=model, method=globalcrit, ncand=ncand, Low=Low, Upp=Upp, ldist=ldist, xstarsloc=xstarsloc)
            afopt["par"] = np.atleast_2d(afopt["par"])

            # Check if the next design is not too close to a TR run result
            if xstarsloc is not None:
                if afopt["value"] == 0 or np.min(np.sqrt(euclidean_dist(afopt["par"], xstarsloc))) < ldist:
                    afits = afits + 1
                    if afits == afitmax and trace > 0:
                        print("Acquisition function optimization failed.")
                    continue
                else:
                    afits = 0

            # # Temporary 2D figures
            # if(plot && d == 2){
            #   Xgrid <- as.matrix(expand.grid(seq(0,1,,101), seq(0,1,,101)))
            #   pgrid <- predict(model, Xgrid)
            #   filled.contour(seq(0,1,,101), seq(0,1,,101), matrix(pgrid$mean, 101), main = "Mean",
            #                  plot.axes = {axis(1);axis(2); points(X); points(xstarsloc, pch = 20, col= "red")})
            #   if(globalcrit == "EIp"){
            #     EIgrid <- apply(Xgrid, 1, crit_EIp, model = model, xstarsloc = xstarsloc, ldist = ldist)
            #     filled.contour(seq(0,1,,101), seq(0,1,,101), matrix(EIgrid, 101), main = "EI",
            #                    plot.axes = {axis(1);axis(2); points(X); points(xstarsloc, pch = 20, col= "red");
            #                      points(afopt$par[1], afopt$par[2], col = "blue", pch = 18)})
            #     }else{
            #     EIgrid <- apply(Xgrid, 1, crit_logEIp, model = model, xstarsloc = xstarsloc, ldist = ldist)
            #     filled.contour(seq(0,1,,101), seq(0,1,,101), matrix(EIgrid, 101), main = "logEI",
            #                    plot.axes = {axis(1);axis(2); points(X); points(xstarsloc, pch = 20, col= "red");
            #                      points(afopt$par[1], afopt$par[2], col = "blue", pch = 18)})
            #   }
            #
            # }

            ## Check to trigger local TR search

            # Case 1) Too close to an existing design
            if tol_dist > 0:
                ## Check if new design is not to close to existing design
                dists = np.sqrt(euclidean_dist(afopt["par"], model["X0"]))
                if np.min(dists) < tol_dist:
                    globalit = False
                    ik = np.argmin(dists)
                    xk = model["X0"][ik, :]
                    if trace > 0:
                        print("EI design too close to existing design, start local search.")
                    continue

            # Case 2) Noise error is stronger than interpolation error
            # Noiseless covariance matrix
            Cm = cov_gen(X1=model["X0"], theta=model["theta"], type=model["covtype"]) + np.diag(model["eps"] * np.ones(model["X0"].shape[0]))
            Cmi = np.linalg.pinv(Cm)
            kx = cov_gen(X1=afopt["par"], X2=model["X0"], theta=model["theta"], type=model["covtype"])
            errinterp = model["nu_hat"] * (1 - kx @ Cmi @ kx.T)

            # Check: px$sd2 should be equal to errinterp + ... see Beek2021

            px = model.predict(afopt["par"])

            if checkinterp and px["sd2"] - errinterp > errinterp:
                globalit = False
                xk = afopt["par"]
                if trace > 0:
                    print("Error driven by noise, start local search.")
                continue

            # If interpolation error is leading, add xnew
            nnewrep = tot_rep_thres(xnew=afopt["par"], model=model, threshold=0.3)
            nnewrep = int(np.maximum(1, np.min([nnewrep, maxrep, nfmax - n])))
            if deter:
                nnewrep = 1

            znew = np.zeros(nnewrep)

            for i in range(nnewrep):
                znew[i] = func_sc(afopt["par"] * (Upp - Low) + Low)

            X = np.vstack((X, afopt["par"].repeat(nnewrep, axis=0)))
            Z = np.concatenate((Z, znew))
            Xall = np.vstack((Xall, np.atleast_2d(afopt["par"]).repeat(nnewrep, axis=0)))
            bvec = np.concatenate((bvec, np.repeat(False, nnewrep)))
            n = n + nnewrep

            # For tracking best solution (can only be at locally estimated solutions if there are any
            mtmp = model.mle(X=X, Z=Z, covtype="Matern5_2", settings=dict(trace=trace), known=dict(beta0=beta0, g=gdeter))
            if not deter and len(localres) > 0:
                Xbestcands = xstarsloc
            else:
                Xbestcands = mtmp["X0"]
            ptmp = mtmp.predict(Xbestcands)
            if not all(Xbestcands[np.argmin(ptmp["mean"]), :] == Xbest[-1, :]):
                Xbest = np.vstack((Xbest, np.atleast_2d(Xbestcands[np.argmin(ptmp["mean"]), :])))
                Zbest = np.concatenate((Zbest, np.atleast_1d(np.min(ptmp["mean"]))))
                nbest = np.concatenate((nbest, np.atleast_1d(n)))
                if trace > 0:
                    print("(Update) #Evals:", n, " Current minimum: ", Xbest[-1, :] * (Upp_o - Low_o) + Low_o, " Estimated mininum: ", Zbest[-1], "\n")

            if afopt["value"] < -600:
                stpc = stpc + 1
            else:
                stpc = 0
            if trace > 0:
                print("Current max EI:", exp(afopt["value"]), "(logEI)", afopt["value"], "\n")
        else:

            # Select designs in the initial TR to start with
            ids = None
            for i in range(model["X0"].shape[0]):
                if np.max(np.abs(model["X0"][i, :] - xk)) < localdelta:
                    if ids is None:
                        ids = np.array([i])
                    else:
                        ids = np.concatenate((ids, np.array([i])))
            if ids is None:
                Xinit = None
                Zinit = None
            else:
                Xinit = model["X0"][ids, :]
                Zinit = model["Z0"][ids]

            tropt = OGPIT(
                func_sc,
                Low=Low,
                Upp=Upp,
                nfmax=np.minimum(nfmax - n, localbudget),
                delta=localdelta,
                maxdelta=maxdelta,
                mindelta=mindelta,
                maxrep=maxrep,
                boots=boots,
                maxnn=maxnn,
                modtype=modtype,
                Xinit=Xinit,
                Zinit=Zinit,
                trace=traceogpit,
                lightreturn=False,
                deter=deter,
                acqtype=localcrit,
            )

            if xstarsloc is None:
                xstarsloc = np.atleast_2d(tropt["par"])
            else:
                xstarsloc = np.vstack((xstarsloc, tropt["par"]))
            if fstarsloc is None:
                fstarsloc = np.atleast_1d(tropt["value"])
            else:
                fstarsloc = np.concatenate((fstarsloc, np.atleast_1d(tropt["value"])))
            xstarsloc_all.update({len(xstarsloc_all): tropt["X"]})
            if fstarsloc_all is None:
                fstarsloc_all = tropt["Z"]
            else:
                fstarsloc_all = np.concatenate((fstarsloc_all, (tropt["Z"])))
            globalit = True
            X = np.vstack((X, np.atleast_2d(tropt["par"]).repeat(np.size(tropt["Xlist"]["Zlist"][tropt["ipar"]]), axis=0)))
            Xall = np.vstack((Xall, tropt["Xall"]))
            bvec = np.concatenate((bvec, tropt["bvec"]))
            Z = np.concatenate((Z, tropt["Xlist"]["Zlist"][tropt["ipar"]]))
            n = n + tropt["nevals"]
            localres.update(tropt)

            if trace > 0:
                print("Local optimum estimated at: ", tropt["par"] * (Upp_o - Low_o) + Low_o, " , estimated value", tropt["value"], " budget used:", tropt["nevals"], "\n")

            # For tracking best solution
            mtmp = model.mle(X=X, Z=Z, covtype="Matern5_2", settings=dict(trace=trace), known=dict(beta0=beta0, g=gdeter))
            if not deter:
                Xbestcands = xstarsloc
            else:
                Xbestcands = mtmp["X0"]
            ptmp = mtmp.predict(Xbestcands)
            if not all(Xbestcands[np.argmin(ptmp["mean"]), :] == Xbest[-1, :]):
                Xbest = np.vstack((Xbest, np.atleast_2d(Xbestcands[np.argmin(ptmp["mean"]), :])))
                Zbest = np.concatenate((Zbest, np.atleast_1d(np.min(ptmp["mean"]))))
                nbest = np.concatenate((nbest, np.atleast_1d(n)))
                if trace > 0:
                    print("(Update) #Evals:", n, " Current minimum: ", Xbest[-1, :] * (Upp_o - Low_o) + Low_o, " Estimated mininum: ", Zbest[-1], "\n")

        if trace > 0 and fstarsloc is not None:
            print("#Evals:", n, " Current minimum: ", xstarsloc[np.argmin(fstarsloc), :] * (Upp_o - Low_o) + Low_o, " Estimated mininum: ", np.min(fstarsloc), "\n")

    if lightreturn:
        return dict(par=xstarsloc[np.argmin(fstarsloc), :] * (Upp_o - Low_o) + Low_o, value=np.min(fstarsloc), localpar=xstarsloc * (Upp_o - Low_o) + Low_o, localvalue=fstarsloc)
    else:
        if xstarsloc is None:
            return dict(
                par=Xbest[-1, :] * (Upp_o - Low_o) + Low_o,
                value=Zbest[-1],
                localpar=None,
                localvalue=None,
                X=X * (Upp_o - Low_o) + Low_o,
                Z=Z,
                localres=localres,
                Xall=Xall * (Upp_o - Low_o) + Low_o,
                bvec=bvec,
                Xbest=Xbest * (Upp_o - Low_o) + Low_o,
                Zbest=Zbest,
                nbest=nbest,
                lstep=lstep,
            )
        else:
            return dict(
                par=xstarsloc[np.argmin(fstarsloc), :] * (Upp_o - Low_o) + Low_o,
                value=np.min(fstarsloc),
                localpar=xstarsloc * (Upp_o - Low_o) + Low_o,
                localvalue=fstarsloc,
                X=X * (Upp_o - Low_o) + Low_o,
                Z=Z,
                localres=localres,
                Xall=Xall * (Upp_o - Low_o) + Low_o,
                bvec=bvec,
                Xbest=Xbest * (Upp_o - Low_o) + Low_o,
                Zbest=Zbest,
                nbest=nbest,
                lstep=lstep,
            )


if __name__ == "__main__":
    d = 3  # Number of variables [2]
    # rbftype = 'cubic'  # Type of RBF (multiquadric, cubic, Gaussian) ['cubic']
    nfmax = 1e5  # 500  # Maximum number of function evaluations per local minimization [60]
    maxrep = 5e2  # Maximum number of evaluation per iteration (multiplied by ns)
    ninit = max(10, 2 * d)  # Number of initial design points
    nmpmax = 2 * d + 1  # Maximum number of model points [2*n+1]
    # gtol = 1e-05  # Gradient tolerance used to stop the local minimization [1e-5]
    gammam = 0.8
    beta = 1e-3
    eta1 = 0.2
    minnn = d + 1
    relvarxnew = 4
    trace = 1
    deter = False
    modtype = "homGP"
    ncand = "small"
    maxnn = 200
    delta = 0.2  # initial trust region radius
    mindelta = 1e-6
    maxdelta = 0.5  # maximum trust region radius
    mintheta = min(0.5, 0.1 * sqrt(d))
    maxtheta = min(5 * sqrt(d), 10)
    vredthrestot = 0.2
    iso = False
    normalize = True
    toldist = 1e-4
    boots = True

    # FIX THE STATE OF THE RANDOM NUMBER GENERATORS TO REPRODUCE RESULTS=======
    randstate = 3
    np.random.seed(randstate)
    # r = rpy2.robjects.r
    # set_seed = r("set.seed")
    # set_seed(int(randstate))

    # Test problem definition
    funcname = jeffsfav  # jeffsfav  # testfcn3  # Name of the function you'd like to minimize ['testfcn']
    functruename = jeffsfav_true  # jeffsfav_true  # testfcn3_true
    Low = -np.ones(d)  # np.zeros(d) # 1-by-n Vector of lower bounds [zeros(1,n)]
    Upp = np.ones(d)  # 1-by-n Vector of upper bounds [ones(1,n)]
    xstars = np.atleast_2d(np.zeros(d))  # np.array([[0.1233668, 0.8192679], [0.5421332, 0.1518330], [0.9608988, 0.1640326]])
    fstars = 0  #  0.3978874

    globalopt = False  # Local or global opt bench?

    acqtype = "EI"  # "costRI" # "qRIauto", "EI"

    if acqtype == "costRI" or acqtype == "costRI2":
        c0 = 1
        c1 = 1
    else:
        c0 = None
        c1 = None

    #  ns = 1  # Replication is handled internally
    # nsmax = 20

    # nfs = 1 # number of initially evaluated points
    # F = np.zeros((nfs,nsmax))
    # F.fill(np.nan)
    # F2 = csr_matrix((nfs, nsmax)) # Warning: sparse matrices contain zero as empty values, it should not influence the results
    # F3 = lil_matrix((nfs, nsmax))

    for noise in [0, 0.0001, 0.001, 0.01, 0.1, 1]:
        print(noise)
        if noise == 0:
            deter = True
        else:
            deter = False
        # Xtmp = np.array([[0.3, 0.4]])  # The points initially evaluated
        # Ftmp = funcname(Xtmp[0])  # Their function values
        # print(Ftmp)
        # Xlist = hetGP.find_reps(Xtmp, Ftmp)

        if globalopt:
            # Global search (in progress)
            gres = STOGPIT(func=funcname, Low=Low, Upp=Upp, nfmax=nfmax, maxnn=maxnn, maxrep=maxrep, trace=1, mindelta=mindelta, localdelta=delta, modtype=modtype, lightreturn=False, deter=deter, maxXu=100)
            print(" (Global opt) Minimum for noise=%f is declared at: " % noise, gres["par"], "Predicted value:", gres["value"], " Value: ", functruename(gres["par"]), "Budget: ", gres["Xall"].shape[0])

            ygnews = np.zeros(gres["Xbest"].shape[0])
            for ii in range(gres["Xbest"].shape[0]):
                ygnews[ii] = functruename(gres["Xbest"][ii, :])

            plt.plot(gres["nbest"], np.log10(ygnews - fstars), "b-")
            plt.title(label="Log10 regret")
            plt.xlabel("Iteration")
            plt.legend(("Center values", "New point values"), loc="upper right")
        else:
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
                acqtype=acqtype,
                c0=c0,
                c1=c1,
            )

            print("(Local opt) Minimum for noise=%f is declared at: " % noise, res["par"], "Predicted value:", res["value"], " Value: ", functruename(res["par"]), "Budget: ", res["nevals"])
            # plt.plot(res[2][:,0], res[2][:,1], 'bo')
            # plt.show()
            sdists = scipy.spatial.distance.cdist(np.atleast_2d(res["par"]), xstars)
            loid = np.argmin(sdists)
            print("Regret:", functruename(res["par"]) - functruename(xstars[loid, :]))
            print(res["Xlist"]["mult"][0])

            # y_tmp = np.zeros(res[2].shape[0])  # To plot all evaluated designs
            # for i in range(res[2].shape[0]):
            #     y_tmp[i] = functruename(res[2][i, :])
            # plt.plot(np.log10(y_tmp - functruename(xstars[loid, :])))
            # plt.show()

            ycenters = np.zeros(res["Xks"].shape[0])
            for ii in range(res["Xks"].shape[0]):
                ycenters[ii] = functruename(res["Xks"][ii, :])
            ynews = np.zeros(res["Xnews"].shape[0])
            for ii in range(res["Xnews"].shape[0]):
                ynews[ii] = functruename(res["Xnews"][ii, :])

            # plt.plot(np.log10(res[3] - testfcn3_true(xstars[loid,:])), 'ro')
            plt.plot(np.log10(ycenters - functruename(xstars[loid, :])))
            plt.plot(np.arange(ynews.shape[0]) + 1, np.log10(ynews - functruename(xstars[loid, :])), "ro")
            if noise > 0:
                plt.plot(np.log10(2 * noise / np.sqrt(res["evalits"])), "g--")
                plt.plot(np.log10(0.675 * noise / np.sqrt(res["evalits"])), "g-")

        plt.show()
