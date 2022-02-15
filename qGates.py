import numpy as np
import random

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qBitStrings as qb
import qAlgorithms

def power(stateOrGate, m):
    assert m > 0, "m must be positive"
    '''Assumes n >= 1. Given an n-qbit gate or state and m >= 1, returns the
    mth tensor power, which is an (n * m)-qbit gate or state. For the sake of
    time and memory, m should be small.'''
    result = stateOrGate
    for _ in range(1,m):
        result = tensor(result, stateOrGate)
    return result

def function(n, m, f):
    '''Assumes n, m >= 1. Given a Python function f : {0, 1}^n -> {0, 1}^m.
    That is, f takes as input an n-bit string and produces as output an m-bit
    string, as defined in qBitStrings.py. Returns the corresponding
    (n + m)-qbit gate F.'''
    
    #iterate through all bit strings of length n and length m, converting to bit strings from ints
    
    F = None
    for a in range(2 ** n):
        for b in range(2 ** m):
            alpha = qb.string(n, a)
            beta = qb.string(m, b)

            first_ket = qu.quantumFromClassic(alpha)
            second_ket = qu.quantumFromClassic(qb.addition(beta, f(alpha)))

            result = qg.tensor(first_ket, second_ket)
            
            column = np.array([result]).T
            #replace this check and the one in tensor since you can concat to an empty array - see simon()
            if F is None:
                F = column
            else:
                F = np.concatenate((F, column), axis = 1)
    return F

def application(u, ketPsi):
    '''Assumes n >=. Applies the n-qbit gate U to the n-qbit state |psi>, returning the n-qbit state U |psi>.'''
    return np.dot(u, ketPsi)

def tensor(a, b):
    '''Assumes that n, m >= 1. Assumes that a is an n-qbit state and b is an
    m-qbit state, or that a is an n-qbit gate and b is an m-qbit gate. Returns
    the tensor product of a and b, which is an (n + m)-qbit gate or state.'''

    #convert a to matrix if not already and make vectors vertical columns
    if len(a.shape) == 1 or len(b.shape) == 1:
        a = np.array([a]).T
        b = np.array([b]).T
    result = None
    nrows = a.shape[0]
    ncols = a.shape[1]
    for r in range(0, nrows):
        row_chunk = a[r,0] * b
        for c in range(1, ncols):
            row_chunk = np.concatenate((row_chunk, a[r,c] * b), axis = 1)
       
        if result is not None:
            result = np.concatenate((result, row_chunk))
        else:
            result = row_chunk

    #return a vector if only one column (and revert make to an array by transposing the column)
    if result.shape[1] == 1:
        result = result.T[0]
    return result

### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))

def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest")
    else:
        print("FAILED tensorTest")
        print("    a = " + str(a))
        print("    b = " + str(b))

def powerTest():
    #unoffical test just to look at some results
    print(power(qc.i, 3))

def functionTest(n, m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]
    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]
    # Build the corresponding gate F.
    ff = function(n, m, f)
    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1
    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket
    # Check 2^n - 1 values somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
        # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here’s F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")

### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    #tensorTest()
    #tensorTest()
    functionTest(1,1)
    functionTest(1,2)
    functionTest(2,1)
    functionTest(3,3)

if __name__ == "__main__":
    main()
