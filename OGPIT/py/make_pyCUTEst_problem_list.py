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

import numpy as np
import ipdb

import pycutest

#########################################
# SET YOUR QUERY PARAMETERS HERE
#########################################

filters = {"constraints": "unconstrained", "regular": True, "n": [2, 15], "userN": True}  # Unconstrained problems

probstrings = pycutest.find_problems(**filters)

nprobs = len(probstrings)

print(f"There are %d problems matching the specifications." % (nprobs))

# ipdb.set_trace(context=21)


## write the query results to a file
# import sys
# sys.stdout = open(r"pyCUTEst_query.txt", "w") # write mode
# for probstr in probstrings:
#    pycutest.print_available_sif_params(probstr)
#    #print("Problem %s has n=%d" % (prob.name, prob.n))


#########################################
# END QUERY
#########################################

from pycutest import import_problem


def make_pyCUTEst_problem_list():

    #    ipdb.set_trace(context=21)

    # Create problem instances
    #    fminsurf = import_problem('FMINSURF', sifParams={'P': 4}) # n=16
    engval1 = import_problem("ENGVAL1", sifParams={"N": 2})
    powellsg = import_problem("POWELLSG", sifParams={"N": 4})
    trigon1 = import_problem("TRIGON1", sifParams={"N": 10})
    #    penalty3 = import_problem('PENALTY3', sifParams={'N/2': 25})
    arglinc = import_problem("ARGLINC", sifParams={"N": 10, "M": 20})  # changed from M=20
    #    ncb20 = import_problem('NCB20', sifParams={'N': 100})
    inteqnels = import_problem("INTEQNELS", sifParams={"N": 10})
    sensors = import_problem("SENSORS", sifParams={"N": 2})
    eigenbls = import_problem("EIGENBLS", sifParams={"N": 2})
    scurly10 = import_problem("SCURLY10", sifParams={"N": 10})
    indefm = import_problem("INDEFM", sifParams={"N": 10})  # letting ALPHA be default 0.5
    dqdrtic = import_problem("DQDRTIC", sifParams={"N": 10})
    #    fminsurf2 = import_problem('FMINSRF2', sifParams={'P': 4}) # n=16
    fletcbv3 = import_problem("FLETCBV3", sifParams={"N": 10, "KAPPA": 0})
    modbeale = import_problem("MODBEALE", sifParams={"N/2": 1, "ALPHA": 50})
    broyndbdls = import_problem("BROYDNBDLS", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 5, "KAPPA3": 1, "LB": 5, "UB": 1})
    sparsqur = import_problem("SPARSQUR", sifParams={"N": 10})
    #    curly30 = import_problem('CURLY30', sifParams={'N': 100})
    #    vandanmsls = import_problem('VANDANMSLS', sifParams={'KNOTS': 20}) # means N=22
    hilberta = import_problem("HILBERTA", sifParams={"N": 2, "D": 0})
    dixmaanf = import_problem("DIXMAANF", sifParams={"M": 5})  # N=15
    #    eigencls = import_problem('EIGENCLS', sifParams={'M': 2}) # N=30
    chainwoo = import_problem("CHAINWOO", sifParams={"NS": 1})
    dixmaanh = import_problem("DIXMAANH", sifParams={"M": 5})  # N=15
    errinros = import_problem("ERRINROS", sifParams={"N": 10})
    strtchdv = import_problem("STRTCHDV", sifParams={"N": 10})
    #    yatp1cls = import_problem('YATP1CLS', sifParams={'N': 10})# n=120
    mancino = import_problem("MANCINO", sifParams={"N": 10, "ALPHA": 5, "BETA": 14, "GAMMA": 3})
    dixon3dq = import_problem("DIXON3DQ", sifParams={"N": 10})
    dixmaano = import_problem("DIXMAANO", sifParams={"M": 5})  # N=15

    print("Done importing first block")

    chnrosnb = import_problem("CHNROSNB", sifParams={"N": 10})
    woods = import_problem("WOODS", sifParams={"NS": 1})
    sinquad2 = import_problem("SINQUAD2", sifParams={"N": 5})
    #    curly20 = import_problem('CURLY20', sifParams={'N': 100})
    #    tenfoldtrls = import_problem('10FOLDTRLS') # n=1000
    arglinb = import_problem("ARGLINB", sifParams={"N": 10, "M": 20})
    morebv = import_problem("MOREBV", sifParams={"N": 10})
    penalty2 = import_problem("PENALTY2", sifParams={"N": 4})
    fletchcr = import_problem("FLETCHCR", sifParams={"N": 10})
    sparsine = import_problem("SPARSINE", sifParams={"N": 10})
    scosine = import_problem("SCOSINE", sifParams={"N": 10})
    #    edensch = import_problem('EDENSCH', sifParams={'N': 36}) # N=36
    dixmaann = import_problem("DIXMAANN", sifParams={"M": 5})  # N=15
    sbrybnd = import_problem("SBRYBND", sifParams={"N": 10})
    dixmaang = import_problem("DIXMAANG", sifParams={"M": 5})  # N=15
    noncvxu2 = import_problem("NONCVXU2", sifParams={"N": 10})
    fletcbv2 = import_problem("FLETCBV2", sifParams={"N": 10, "KAPPA": 1})
    eigenals = import_problem("EIGENALS", sifParams={"N": 2})
    #    liarwhd = import_problem('LIARWHD', sifParams={'N': 36}) # N=36
    broydn7d = import_problem("BROYDN7D", sifParams={"N/2": 5})
    dixmaanj = import_problem("DIXMAANJ", sifParams={"M": 5})  # N=15

    print("Done importing second block")

    broydn3dls = import_problem("BROYDN3DLS", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 1})
    #    scurly20 = import_problem('SCURLY20', sifParams={'N': 10}) # causes error???
    #    quartc = import_problem('QUARTC', sifParams={'N': 25}) # N=25
    dixmaanm1 = import_problem("DIXMAANM1", sifParams={"M": 5})  # N=15
    fletbv3m = import_problem("FLETBV3M", sifParams={"N": 10, "KAPPA": 1})
    kssls = import_problem("KSSLS", sifParams={"N": 4})
    box = import_problem("BOX", sifParams={"N": 10})
    #    luksan17ls = import_problem('LUKSAN17LS')  # This problem doesn't have adjustable parameters
    dixmaanc = import_problem("DIXMAANC", sifParams={"M": 5})  # N=15
    dixmaand = import_problem("DIXMAAND", sifParams={"M": 5})  # N=15
    sscosine = import_problem("SSCOSINE", sifParams={"N": 10})
    msqrtals = import_problem("MSQRTALS", sifParams={"P": 2})  # N=4
    powersum = import_problem("POWERSUM", sifParams={"N": 4})
    testquad = import_problem("TESTQUAD", sifParams={"N": 10})
    dixmaani1 = import_problem("DIXMAANI1", sifParams={"M": 5})  # N=15
    #    spmsrtls = import_problem('SPMSRTLS', sifParams={'M': 10})# N=28
    #    vareigvl = import_problem('VAREIGVL', sifParams={'N': 19, 'M': 6}) #default Q=1.5 # N=20
    #    spinls = import_problem('SPINLS', sifParams={'N': 2})# matrix dimension? # gives an Inf

    print("Done importing third block")

    brybnd = import_problem("BRYBND", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 5, "KAPPA3": 1, "LB": 5, "UB": 1})
    oscipath = import_problem("OSCIPATH", sifParams={"N": 2, "RHO": 1})
    vardim = import_problem("VARDIM", sifParams={"N": 10})
    tointgss = import_problem("TOINTGSS", sifParams={"N": 10})
    #    bdqrtic = import_problem('BDQRTIC', sifParams={'N': 100})
    #    yatp1ls = import_problem('YATP1LS', sifParams={'N': 10})# n=120
    cosine = import_problem("COSINE", sifParams={"N": 10})
    #    ncb20b = import_problem('NCB20B', sifParams={'N': 21}) # N=21
    spin2ls = import_problem("SPIN2LS", sifParams={"N": 2})  # matrix dimension?
    dixmaane1 = import_problem("DIXMAANE1", sifParams={"M": 5})  # N=15
    #    luksan16ls = import_problem('LUKSAN16LS')
    arglina = import_problem("ARGLINA", sifParams={"N": 10, "M": 20})
    nonmsqrt = import_problem("NONMSQRT", sifParams={"P": 3})  # N=9
    ssbrybnd = import_problem("SSBRYBND", sifParams={"N": 10})
    noncvxun = import_problem("NONCVXUN", sifParams={"N": 10})
    #    luksan21ls = import_problem('LUKSAN21LS')
    argtrigls = import_problem("ARGTRIGLS", sifParams={"N": 10})
    dixmaana1 = import_problem("DIXMAANA1", sifParams={"M": 5})  # N=15
    penalty1 = import_problem("PENALTY1", sifParams={"N": 4})
    yatp2cls = import_problem("YATP2CLS", sifParams={"N": 2})  # n=8
    yatp2ls = import_problem("YATP2LS", sifParams={"N": 2})  # n=8
    dixmaanp = import_problem("DIXMAANP", sifParams={"M": 5})  # N=15
    power = import_problem("POWER", sifParams={"N": 10})
    sinquad = import_problem("SINQUAD", sifParams={"N": 5})
    srosenbr = import_problem("SROSENBR", sifParams={"N/2": 5})
    indef = import_problem("INDEF", sifParams={"N": 10})  # using default ALPHA=0.5
    tquartic = import_problem("TQUARTIC", sifParams={"N": 5})
    dqrtic = import_problem("DQRTIC", sifParams={"N": 10})
    #    luksan15ls = import_problem('LUKSAN15LS')
    #    cycloocfls = import_problem('CYCLOOCFLS', sifParams={'P': 8}) # N=20
    schmvet = import_problem("SCHMVETT", sifParams={"N": 3})
    oscigrad = import_problem("OSCIGRAD", sifParams={"N": 2, "RHO": 1})
    extrosnb = import_problem("EXTROSNB", sifParams={"N": 5})
    #    qing = import_problem('QING', sifParams={'N': 100})
    #    nondquar = import_problem('NONDQUAR', sifParams={'N': 100})
    hilbertb = import_problem("HILBERTB", sifParams={"N": 5, "D": 5})
    dixmaanb = import_problem("DIXMAANB", sifParams={"M": 5})  # N=15

    print("Done importing fourth block")

    boxpower = import_problem("BOXPOWER", sifParams={"N": 10})
    fletchbv = import_problem("FLETCHBV", sifParams={"N": 10, "KAPPA": 1})
    dixmaanl = import_problem("DIXMAANL", sifParams={"M": 5})  # N=15
    #    arwhead = import_problem('ARWHEAD', sifParams={'N': 100})
    nondia = import_problem("NONDIA", sifParams={"N": 10})
    #    luksan22ls = import_problem('LUKSAN22LS')
    dixmaank = import_problem("DIXMAANK", sifParams={"M": 5})  # N=15
    tridia = import_problem("TRIDIA", sifParams={"N": 10})
    watson = import_problem("WATSON", sifParams={"N": 12})
    trigon2 = import_problem("TRIGON2", sifParams={"N": 10})
    #    scurly30 = import_problem('SCURLY30', sifParams={'N': 10})# causes error???
    freuroth = import_problem("FREUROTH", sifParams={"N": 2})
    genhumps = import_problem("GENHUMPS", sifParams={"N": 5})
    genrose = import_problem("GENROSE", sifParams={"N": 5})
    cragglvy = import_problem("CRAGGLVY", sifParams={"M": 1})  # N=4
    chnrsnbm = import_problem("CHNRSNBM", sifParams={"N": 10})
    #    luksan14ls = import_problem('LUKSAN14LS')
    #    curly10 = import_problem('CURLY10', sifParams={'N': 100})
    msqrtbls = import_problem("MSQRTBLS", sifParams={"P": 3})  # N=9
    errinrsm = import_problem("ERRINRSM", sifParams={"N": 10})
    brownal = import_problem("BROWNAL", sifParams={"N": 10})

    print("Done importing final block")

    # Create list of problem instances
    problems = [
        #        fminsurf,
        engval1,
        powellsg,
        trigon1,
        #        penalty3,
        arglinc,
        #        ncb20,
        inteqnels,
        sensors,
        eigenbls,
        scurly10,
        indefm,
        dqdrtic,
        #        fminsurf2,
        fletcbv3,
        modbeale,
        broyndbdls,
        sparsqur,
        #        curly30,
        #        vandanmsls,
        hilberta,
        dixmaanf,
        #        eigencls,
        chainwoo,
        dixmaanh,
        errinros,
        strtchdv,
        #        yatp1cls,
        mancino,
        dixon3dq,
        dixmaano,
        chnrosnb,
        woods,
        sinquad2,
        #        curly20,
        #        tenfoldtrls,
        arglinb,
        morebv,
        penalty2,
        fletchcr,
        sparsine,
        scosine,
        #        edensch,
        dixmaann,
        sbrybnd,
        dixmaang,
        noncvxu2,
        fletcbv2,
        eigenals,
        #        liarwhd,
        broydn7d,
        dixmaanj,
        broydn3dls,
        #        scurly20,
        #        quartc,
        dixmaanm1,
        fletbv3m,
        kssls,
        box,
        #        luksan17ls,
        dixmaanc,
        dixmaand,
        sscosine,
        msqrtals,
        powersum,
        testquad,
        dixmaani1,
        #        spmsrtls,
        #        vareigvl,
        #        spinls,
        brybnd,
        oscipath,
        vardim,
        tointgss,
        #        bdqrtic,
        #        yatp1ls,
        cosine,
        #        ncb20b,
        spin2ls,
        dixmaane1,
        #        luksan16ls,
        arglina,
        nonmsqrt,
        ssbrybnd,
        noncvxun,
        #        luksan21ls,
        argtrigls,
        dixmaana1,
        penalty1,
        yatp2cls,
        yatp2ls,
        dixmaanp,
        power,
        sinquad,
        srosenbr,
        indef,
        tquartic,
        dqrtic,
        #        luksan15ls,
        #        cycloocfls,
        schmvet,
        oscigrad,
        extrosnb,
        #        qing,
        #        nondquar,
        hilbertb,
        dixmaanb,
        boxpower,
        fletchbv,
        dixmaanl,
        #        arwhead,
        nondia,
        #        luksan22ls,
        dixmaank,
        tridia,
        watson,
        trigon2,
        #        scurly30,
        freuroth,
        genhumps,
        genrose,
        cragglvy,
        chnrsnbm,
        #        luksan14ls,
        #        curly10,
        msqrtbls,
        errinrsm,
        brownal,
    ]

    return problems
