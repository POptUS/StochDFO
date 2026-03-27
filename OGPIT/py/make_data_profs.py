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

import os
import sys
from pickletools import optimize

import matplotlib.pyplot as plt
import numpy as np
from plot_data_profile import plot_data_profile

# Appendix
# A) ## check influence of gamma
# solvers = ["defaultnew", "gamma09", "gamma06"]
# solverscap = ["default (gamma=0.8)", "gamma=0.9", "gamma=0.6"]
## B) check influence of imse coefficient
# solvers = ["defaultnew", "imse01", "imse1", "imse100"]
# solverscap = ["default (imse=10)", "imse=0.1", "imse=1", "imse=100"]
## C) Compare with actual fixed rep
# solvers = ["defaultnew", "fixed250rep", "fixed500rep", "fixed1000rep"]
# solverscap = ["default",  "fixed 250 reps", "fixed 500 reps", "fixed 1000rep"]
## D) check the influence of vred
# solvers = ["defaultnew", "vred05", "vred01"]
# solverscap = ["default (0.2)", "Ta = 0.5", "Ta 0.1"]
## E) check the influence of relvarxnew
solvers = ["defaultnew", "cvred1", "cvred9", "cvred16"]
solverscap = ["default (4)", "PVR 1", "PVR 9", "PVR 16"]


## for results in the paper
# solvers = ["default", "turbo", "botorch", "snowpac"]
# solverscap = ["OGPIT", "TuRBO", "BoTorch", "SNOWPAC"]


## Bendfo results
bendfo = False
# solvers = ["pydefault", "turbo", "botorch"]
# solverscap = ["OGPIT", "TuRBO", "BoTorch"]


# Use regret at center or just evaluated points (naive)
naive = False

if bendfo:
    # Original bendfo problems features
    dims = np.array([9, 9, 7, 7, 7, 7, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 3, 6, 6, 9, 9, 12, 12, 3, 2, 4, 4, 6, 7, 8, 9, 10, 11, 10, 5, 11, 11, 8, 10, 11, 12, 5, 6, 8, 5, 5, 8, 10, 12, 12, 8, 8])
    opt_Problems = np.arange(0, dims.shape[0])
    nrep = 30
    noises = np.atleast_1d(np.array((0.0, 0.001, 0.10, 10)))  # Noise std
else:
    opt_Problems = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # [1,2,3,4,5,6,7,8,9]
    dims = np.array([2, 4, 6, 2, 4, 6, 2, 2, 4])  # [2,4,6,2,4,6,2,2,4]
    nrep = 30  # 40
    noises = np.atleast_1d(np.array((0.0, 0.001, 0.010, 0.100)))  # Noise std


budgets = (1e4 * (dims + 1)).astype("int")  # 5e4 # 1e5 #1e5 #500000 #100000


showfig = False
# results_directory = '/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python/benchmark_results/'    #'./benchmark_results/'
results_directory = "/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python/sens_results/"  #'./benchmark_results/'
if bendfo:
    py_results_directory = "/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python/benchmark_results_bendfo/"
else:
    py_results_directory = "/home/mbinois/Documents/GitProjects/bacasable/Misc/KSP/python/benchmark_results_py/"  #  # '/user/mbinois/home/Documents/GitProjects/bacasable/Misc/KSP/python/benchmark_results_py/'

num_total_probs = len(opt_Problems) * nrep  # *len(noises)

for nois in noises:
    N = np.zeros(num_total_probs)
    FHIST = np.inf * np.ones((np.max(budgets), num_total_probs, len(solvers)))
    for solver_num, s in enumerate(solvers):
        p_count = -1
        for prob_num in opt_Problems:
            if bendfo and (solvers[solver_num] == "pydefault" or solvers[solver_num] == "cma" or solvers[solver_num] == "cman" or solvers[solver_num] == "turbo" or solvers[solver_num] == "botorch"):
                budget = budgets[prob_num]
            else:
                budget = budgets[prob_num - 1]
            xpstmp = np.sort(np.concatenate((np.linspace(10, 90, 9), np.linspace(100, 1000, 10), np.linspace(0, budget, 201)))).astype(int)
            xpstmp[0] = 1
            for rep in np.arange(nrep) + 1:
                if bendfo and (prob_num == 17 or 22 <= prob_num <= 25 or 32 <= prob_num <= 37 or 39 <= prob_num <= 41 or 48 <= prob_num <= 50):
                    continue
                p_count += 1
                if nois == 0:
                    tmpnois = 0
                else:
                    tmpnois = nois

                if solvers[solver_num] == "pydefault":
                    if bendfo:
                        outfilename1 = f"pydefault_bendfo_row={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    else:
                        outfilename1 = f"pydefault_probname={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    Results = np.load(py_results_directory + outfilename1, allow_pickle=True).item()
                    Fvals = np.inf * np.ones((budget))
                    # if bendfo:
                    #     Fvals[np.arange(Results["naiveregret"].shape[0])] = Results["naiveregret"]
                    # else:
                    if naive:
                        Fvals[np.arange(Results["naiveregret"].shape[0])] = Results["naiveregret"]
                    else:
                        Fvals[xpstmp - 1] = Results["regretatxps"]

                elif solvers[solver_num] == "cma" or solvers[solver_num] == "cman" or solvers[solver_num] == "turbo" or solvers[solver_num] == "botorch":
                    if solvers[solver_num] == "cma":
                        outfilename1 = f"cma_bendfo_row={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    if solvers[solver_num] == "cman":
                        if bendfo:
                            outfilename1 = f"cman_bendfo_row={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                        else:
                            outfilename1 = f"cman_probname={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    if solvers[solver_num] == "turbo":
                        if bendfo:
                            outfilename1 = f"turbo_bendfo_row={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                        else:
                            outfilename1 = f"turbo_probname={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    if solvers[solver_num] == "botorch":
                        if bendfo:
                            outfilename1 = f"botorch_bendfo_row={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                        else:
                            outfilename1 = f"botorch_probname={prob_num}_nfmax={budget}_noise={nois}_seed={rep - 1}_regret_and_naiveregret.npy"
                    Results = np.load(py_results_directory + outfilename1, allow_pickle=True).item()
                    Fvals = np.inf * np.ones((budget))
                    # Fvals[np.arange(Results["naiveregret"].shape[0])] = Results["naiveregret"]

                    if naive:
                        Fvals[np.arange(Results["naiveregret"].shape[0])] = Results["naiveregret"]
                    else:
                        tmp = Results["regretatxps"]
                        if np.isnan(tmp[0]):  # for cma, somehow the first value is nan
                            tmp[0] = tmp[1]
                        Fvals[xpstmp[np.arange(len(tmp))] - 1] = tmp

                else:
                    outfilename1 = f"{solvers[solver_num]}_probname={prob_num}_nfmax={budget}_noise={tmpnois}_seed={rep}_regret_and_naiveregret.npy"
                    Results = np.load(results_directory + outfilename1, allow_pickle=True).item()
                    Fvals = np.inf * np.ones((budget))
                    if naive:
                        Fvals[np.arange(Results["naiveregret"].shape[0])] = Results["naiveregret"]
                    else:
                        Fvals[np.array(Results["xps"]).astype(int) - 1] = Results["regretatxps"]

                if solver_num == 0:
                    if bendfo:
                        n = dims[prob_num]
                    else:
                        n = dims[prob_num - 1]  # Results['X'].shape[1]
                    N[p_count] = n + 1

                FHIST[0 : len(Fvals), p_count, solver_num] = Fvals

    if bendfo:
        if p_count + 1 < num_total_probs:
            FHIST = FHIST[:, np.arange(p_count + 1), :]
            N = N[np.arange(p_count + 1)]
        # Eventually to filter some problems
        # ids = np.where(FHIST[0,:,0] < 1e4)
        # FHIST = FHIST[:,ids[0],:]
        # N = N[ids[0]]

    ngate = 7
    # plt.rcParams['figure.figsize'] = [5, 4]
    plt.rcParams.update({"font.size": 16})
    for gate in np.array([1e-6, 1e-3, 1e-1]):  # np.logspace(-ngate, -1, 4):
        plot_data_profile(FHIST, N, gate, optimality_type="value", legendstr=solverscap)
        plt.title(f"gate = {gate:1.1e}, noise {nois}")
        plt.xlabel("Number of Function Evaluations / (d+1)")  # overwrite xlabel
        # plt.tight_layout()
        plt.savefig(f"perf_prof_gate={gate:1.1e}_noise_{nois}.png", dpi=300)
        if showfig:
            plt.show()
        plt.close()

    plot_progress = False
    if plot_progress:
        for j in range(FHIST.shape[2]):
            for i in range(1, FHIST.shape[0]):
                FHIST[i, :, j] = np.minimum(FHIST[i, :, j], FHIST[i - 1, :, j])
        if bendfo:
            fig, axs = plt.subplots(6, 6, figsize=(20, 20), sharey=True)
            nbf = 6
        else:
            fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharey=True)
            nbf = 3
        fig.suptitle(f"progress_noise={nois}")
        for i in np.arange(FHIST.shape[1]):
            if i % nrep == 0:
                pbnum = i // nrep
                color = iter(plt.cm.rainbow(np.linspace(0, 1, FHIST.shape[2])))
                curm = np.min(FHIST[:, np.arange(i, i + nrep), :])
                if not naive:
                    curm = 0
                tmp = np.median(FHIST[:, np.arange(i, i + nrep), :], axis=1)
                tmp2 = np.min(FHIST[:, np.arange(i, i + nrep), :], axis=1)
                tmp3 = np.max(FHIST[:, np.arange(i, i + nrep), :], axis=1)
                for j in np.arange(FHIST.shape[2]):
                    c = next(color)
                    axs[pbnum // nbf, pbnum % nbf].plot(np.log10(tmp[:, j] - curm + 1e-15), c=c, label=solvers[j])
                    axs[pbnum // nbf, pbnum % nbf].plot(np.log10(tmp2[:, j] - curm + 1e-15), "--", c=c)
                    axs[pbnum // nbf, pbnum % nbf].plot(np.log10(tmp3[:, j] - curm + 1e-15), "--", c=c)
                axs[pbnum // nbf, pbnum % nbf].legend()
                axs[pbnum // nbf, pbnum % nbf].set_title("Problem" + str(i // nrep))
        plt.tight_layout()
        if showfig:
            plt.show(block=True)

        plt.savefig(f"progress_gate={gate:1.1e}_noise_{nois}.png", dpi=300)
        plt.close()
        # for i in np.arange(FHIST.shape[1]):
        #     if i % nrep == 0:
        #         color = iter(plt.cm.rainbow(np.linspace(0, 1, FHIST.shape[2])))
        #         curm = np.min(FHIST[:,np.arange(i, i+nrep),:])
        #         tmp = np.mean(FHIST[:,np.arange(i, i+nrep),:], axis=1)
        #         tmp2 = np.min(FHIST[:, np.arange(i, i + nrep), :], axis=1)
        #         for j in np.arange(FHIST.shape[2]):
        #             c = next(color)
        #             plt.plot(np.log10(tmp[:,j] - curm + 1e-16), c=c, label=solvers[j])
        #             plt.plot(np.log10(tmp2[:,j] - curm + 1e-16),"--", c=c)
        #         plt.legend()
        #         plt.title("Problem" + str(i//nrep))
        #         plt.show(block=True)
