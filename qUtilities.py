import math
import random
import numpy

import qConstants as qc
import qBitStrings as qb


def equal(a, b, epsilon):
    '''Assumes that n >= 0. Assumes that a and b are both n-qbit states or n-qbit gates. Assumes that epsilon is a positive (but usually small) real number. Returns whether a == b to within a tolerance of epsilon. Useful for doing equality comparisons in the floating-point context. Warning: Does not consider global phase changes; for example, two states that are global phase changes of each other may be judged unequal. Warning: Use this function sparingly, for inspecting output and running tests. Probably you should not use it to make a crucial decision in the middle of a big computation. In past versions of CS 358, this function has not existed. I have added it this time just to streamline the tests.'''
    diff = a - b
    if len(diff.shape) == 0:
        # n == 0. Whether they're gates or states, a and b are scalars.
        return abs(diff) < epsilon
    elif len(diff.shape) == 1:
        # a and b are states.
        return sum(abs(diff)) < epsilon
    else:
        # a and b are gates.
        return sum(sum(abs(diff))) < epsilon

def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return qc.one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            psi = numpy.array([reals[i] + imags[i] * 1j for i in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm

def bitValue(state):
    '''Given a one-qbit state assumed to be exactly classical --- usually because a classical state was just explicitly assigned to it --- returns the corresponding bit value 0 or 1.'''
    if (state == qc.ket0).all():
        return 0
    else:
        return 1

def powerMod(k, l, m):
    '''Given non-negative integer k, non-negative integer l, and positive integer m. Computes k^l mod m. Returns an integer in {0, ..., m - 1}.'''
    kToTheL = 1
    curr = k
    while l >= 1:
        if l % 2 == 1:
            kToTheL = (kToTheL * curr) % m
        l = l // 2
        curr = (curr * curr) % m
    return kToTheL

def quantumFromClassic(bitstring):
    n = len(bitstring)

    arr = numpy.array((0,) * (2 ** n))

    one_position = qb.integer(bitstring)

    arr[one_position] = 1

    return arr

def quantumListToClassicTuple(quantum_list):
    arr = []
    for qbit in quantum_list:
        arr.append(bitValue(qbit))

    return tuple(arr)

def removeZeroRow(arr):
    width = len(arr[0])
    zero_row = (0,) * width
   
    copy = list(arr)
    if copy[-1] == zero_row:
        copy.pop()

    return copy

def missingLeadingRow(arr):
    width = len(arr[0])
    one_row = [1,] * width

    copy = list(arr)
    for row in arr:
        first_one_index = row.index(1)
        one_row[first_one_index] = 0

    return one_row

def continuedFraction(n, m, x0):
    '''x0 is a float in [0, 1). Tries probing depths j = 0, 1, 2, ... until
    the resulting rational approximation x0 ~ c / d satisfies either d >= m or
    |x0 - c / d| <= 1 / 2^(n + 1). Returns a pair (c, d) with gcd(c, d) = 1.'''
    j = 0
    while True:
        c, d = fraction(x0, j)
        
        if d >= m or abs(x0 - (c / d)) <= (1 / 2 ** (n + 1)):
            return (c, d)
        
        j += 1

def fraction(x0, j):
    '''recursive helper to continuedFraction
    calculates the c and d out j times to approxiate x0 as c / d'''
    
    if x0 == 0:
        return (0, 1)
    elif j == 0:
        a0 = math.floor(1 / x0)
        c = 1
        d = a0
        return (c, d)
    else:
        a0 = math.floor(1 / x0)
        x1 = (1 / x0) - a0
        next_c, next_d = fraction(x1, j - 1)

        c = next_d
        d = (a0 * next_d) + next_c

        return (c, d)

def fractions_test():
    #test fraction
    print(fraction(1 / math.pi, 2))
    print(fraction(1 / 7, 4))
    print(fraction(0, 3))

    print(continuedFraction( 8, 4, (1 / math.pi)))


if __name__ == "__main__":
    fractions_test()
    