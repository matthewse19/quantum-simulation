


import random
import math
import numpy

import qConstants as qc
import qUtilities as qu
import qGates as qg



def first(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the first qbit. Returns a pair (tuple of two items) consisting of a classical one-qbit state (either |0> or |1>) and an (n - 1)-qbit state.'''
    state_size = len(state)
    squared_entries = [numpy.abs(entry) ** 2 for entry in state]

    top_state = state[:state_size//2,]
    bottom_state = state[state_size//2:,]

    sigma0 = math.sqrt(numpy.sum(squared_entries[:state_size//2]))
    sigma1 = math.sqrt(numpy.sum(squared_entries[state_size//2:]))

    if sigma0 == 0:
        ketphi = bottom_state
        return (qc.ket1, ketphi)
    elif sigma1 == 0:
        ketchi = top_state
        return (qc.ket0, ketchi)
    else:
        ketchi = (1 / sigma0) * top_state
        ketphi = (1 / sigma1) * bottom_state
        if random.random() < sigma0 ** 2:
            return (qc.ket0, ketchi)
        else:
            return (qc.ket1, ketphi)


def last(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the last qbit. Returns a pair consisting of an (n - 1)-qbit state and a classical 1-qbit state (either |0> or |1>).'''
    state_size = len(state)
    squared_entries = [numpy.abs(entry) ** 2 for entry in state]
    
    #every other entry starting at 0
    first_state = state[::2]
    #every other entry starting at 1
    second_state = state[1::2]

    sigma0 = math.sqrt(numpy.sum(squared_entries[::2]))
    sigma1 = math.sqrt(numpy.sum(squared_entries[1::2]))

    if sigma0 == 0:
        ketphi = second_state
        return (ketphi, qc.ket1)
    elif sigma1 == 0:
        ketchi = first_state
        return (ketchi, qc.ket0)
    else:
        ketchi = (1 / sigma0) * first_state
        ketphi = (1 / sigma1) * second_state
        if random.random() < sigma0 ** 2:
            return (ketchi, qc.ket0)
        else:
            return (ketphi, qc.ket1)

### DEFINING SOME TESTS ###

def firstTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |0> |psi> or |1> |psi>, measures the first qbit, and then reconstructs the state.
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket0, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest first part")
    else:
        print("failed firstTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket1, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest second part")
    else:
        print("failed firstTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def firstTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(qc.ket0, ketChi) + psi1 * qg.tensor(qc.ket1, ketPhi)
    def f():
        if (first(ketOmega)[0] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check firstTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))

def lastTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |psi> |0> or |psi> |1>, measures the last qbit, and then reconstructs the state.
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket0)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest first part")
    else:
        print("failed lastTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket1)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest second part")
    else:
        print("failed lastTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def lastTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(ketChi, qc.ket0) + psi1 * qg.tensor(ketPhi, qc.ket1)
    def f():
        if (last(ketOmega)[1] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check lastTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))



### RUNNING THE TESTS ###

def main():
    n = 4
    firstTest(n)
    firstTest(n)
    firstTest345(n, 1000)
    firstTest345(n, 1000)
    lastTest(n)
    lastTest(n)
    lastTest345(n, 1000)
    lastTest345(n, 1000)

if __name__ == "__main__":
    main()


