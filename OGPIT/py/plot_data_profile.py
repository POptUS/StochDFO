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

import ipdb
import matplotlib.pyplot as plt
import numpy as np


def plot_data_profile(HIST, N, gate, optimality_type="value", legendstr=None):
    """
    This subroutine produces a data profile as described in:

    Benchmarking Derivative-Free Optimization Algorithms
    Jorge J. More' and Stefan M. Wild
    SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.

    The subroutine returns a handle to lines in a data profile.

      HIST contains a three dimensional array of function values.
        H[f,p,s] = function value # f for problem p and solver s.
      N is an np-vector of (positive) budget units. If simplex
        gradients are desired, then N(p) would be n(p)+1, where n(p) is
        the number of variables for problem p.
      gate is a positive constant reflecting the convergence tolerance.
           meaning is interpreted by the optimality type
      optimality_type is a string indicating the measure of optimality to use
           "value"   -> use smallest function value seen by all solvers
           "absgrad" -> solved when absolute gradient is small, ||nabla f_k|| <= gate
           "relgrad" -> solved when relative gradient is small, ||nabla f_k|| <= gate*||nabla f_0||

    Argonne National Laboratory
    Jorge More' and Stefan Wild. January 2008.
    """

    # ipdb.set_trace(context=21)

    nf, nprob, ns = HIST.shape  # Grab the dimensions

    # Produce a suitable history array with sorted entries:
    for j in range(ns):
        for i in range(1, nf):
            HIST[i, :, j] = np.minimum(HIST[i, :, j], HIST[i - 1, :, j])

    # ipdb.set_trace(context=21)

    match optimality_type:

        case "value":

            prob_min = np.nanmin(HIST, axis=(0, 2))  # The minimum value seen for each problem
            prob_max = HIST[0, :, 0]  # The starting value for each problem

            # For each problem and solver, determine the number of
            # N-function bundles (e.g.- gradients) required to reach the cutoff value
            T = np.zeros((nprob, ns))
            for p in range(nprob):
                cutoff = prob_min[p] + gate * (prob_max[p] - prob_min[p])
                for s in range(ns):
                    nfevs = np.argmax(HIST[:, p, s] <= cutoff)  # use argmax to find first occurrence
                    if nfevs == 0:
                        T[p, s] = np.nan
                    else:
                        T[p, s] = nfevs / N[p]

        case "absgrad":

            # prob_max = HIST[0, :, 0]            # The starting grad norm for each problem

            # For each problem and solver, determine the number of
            # N-function bundles (e.g.- gradients) required to reach the cutoff value
            T = np.zeros((nprob, ns))
            for p in range(nprob):
                cutoff = gate
                for s in range(ns):
                    nfevs = np.argmax(HIST[:, p, s] <= cutoff)  # use argmax to find first occurrence
                    if nfevs == 0:
                        T[p, s] = np.nan
                    else:
                        T[p, s] = nfevs / N[p]

        case "relgrad":

            prob_max = HIST[0, :, 0]  # The starting grad norm for each problem

            # For each problem and solver, determine the number of
            # N-function bundles (e.g.- gradients) required to reach the cutoff value
            T = np.zeros((nprob, ns))
            for p in range(nprob):
                cutoff = gate * prob_max[p]
                for s in range(ns):
                    nfevs = np.argmax(HIST[:, p, s] <= cutoff)  # use argmax to find first occurrence
                    if nfevs == 0:
                        T[p, s] = np.nan
                    else:
                        T[p, s] = nfevs / N[p]

    T += 1  # Need to add one for python indexing.

    ##############################################################
    # plot
    ##############################################################

    # Other colors, lines, and markers are easily possible:
    colors = ["b", "r", "k", "m", "c", "g", "y"]
    lines = ["-", "-.", "--"]
    markers = ["s", "o", "^", "v", "p", "<", "x", "h", "+", "d", "*", "<"]
    # plt.rcParams['figure.figsize'] = [5, 4]
    # ipdb.set_trace(context=21)

    # Replace all NaN's with twice the max_ratio and sort.
    max_data = np.nanmax(T)
    T[np.isnan(T)] = 2 * max_data
    T = np.sort(T, axis=0)
    plt.figure(figsize=(6,5))
    # For each solver, plot stair graphs with markers.
    hl = [None] * ns
    for s in range(ns):
        xs, ys = np.append(T[:, s], T[-1, s]), np.arange(1, nprob + 2) / nprob
        sl = s % 3
        sc = s % 7
        sm = s % 12
        fstring = f"{lines[sl]}{colors[sc]}{markers[sm]}"

        (hl[s],) = plt.step(xs, ys, fstring, where="post", label=legendstr[s])

    plt.axis([0, 1.1 * max_data, 0, 1])
    plt.xlabel("Normalized Iterations")
    plt.ylabel("Proportion of Solved Problems")
    plt.legend()
    # plt.show()

    return hl
