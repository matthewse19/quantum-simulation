import math
import random
import numpy

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm
import qBitStrings as qb


def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992). Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    #A picks |alpha>
    if random.random() < 0.5:
        ketAlpha = qc.ket0
    else:
        ketAlpha = qc.ket1

    #A sets |psi> and sends it to B
    if (ketAlpha == qc.ket0).all():
        ketPsi = qc.ket0
    else:
        ketPsi = qc.ketPlus

    #B picks |beta>
    if random.random() < 0.5:
        ketBeta = qc.ket0
    else:
        ketBeta = qc.ket1

    #B may apply H to |psi>, then measures it
    if (ketBeta == qc.ket0).all():
        ketGamma = qm.first(qg.tensor(ketPsi, qc.ket0))[0]
    else:
        ketGamma = qm.first(qg.tensor(qg.application(qc.h, ketPsi), qc.ket0))[0]

    return (ketAlpha, ketBeta, ketGamma)

def deutsch(f):
    '''Implements the algorithm of Deutsch (1985). That is, given a two-qbit gate F representing a function f : {0, 1} -> {0, 1}, returns |1> if f is constant, and |0> if f is not constant.'''
    
    input = qg.tensor(qc.ket1, qc.ket1)
    
    hh = qg.tensor(qc.h, qc.h)
    matrixF = f
    circuit = numpy.matmul(hh, numpy.matmul(matrixF, hh))

    output = qg.application(circuit, input)

    return qm.first(output)[0]

def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate F representing a function
    f : {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown delta
    in {0, 1}^n, returns the list or tuple of n classical one-qbit states (each
    |0> or |1>) corresponding to delta.'''
    h_chunk = qg.power(qc.h, n + 1)
    circuit = numpy.matmul(h_chunk, numpy.matmul(f, h_chunk))

    input = qg.tensor(qg.power(qc.ket0, n), qc.ket1)
    output = qg.application(circuit, input)

    #measure first n qbits

    leftover_measurement = output
    measurement = []
    while len(measurement) < n:
        first = qm.first(leftover_measurement)
        first_measurement = first[0]
        leftover_measurement = first[1]
        measurement.append(first_measurement)
    
    return measurement

def simon(n, f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F
    representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit
    string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly
    random bit string gamma that is perpendicular to delta.'''
    input = qg.power(qc.ket0, n + n - 1)

    small_h_chunk = qg.power(qc.h, n)
    large_h_chunk = qg.tensor(small_h_chunk, qg.power(qc.i, n - 1))
    first_circuit = numpy.matmul(f, large_h_chunk)

    first_output = qg.application(first_circuit, input)

    leftover_measurement = first_output
    measurement = []
    while len(measurement) < n - 1:
        leftover_measurement, last_measurement = qm.last(leftover_measurement)
        measurement.append(last_measurement)
    
    second_output = qg.application(small_h_chunk, leftover_measurement)

    leftover_measurement = second_output
    measurement = []
    while len(measurement) < n:
        first_measurement, leftover_measurement = qm.first(leftover_measurement)
        measurement.append(first_measurement)
    
    return measurement

def fourier(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.'''
    t = numpy.zeros((2 ** n, 2 ** n), dtype=numpy.array(0 + 0j).dtype)

    #rows
    for a in range(2 ** n):
        #columns
        for b in range(2 ** n):
            t[a, b] = (1/ (2 ** (n/2))) * numpy.exp((0 + 1j) * 2 * numpy.pi * a * b * (1 / (2 ** n)))

    return t

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate F representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple
    of n classical one-qbit states (|0> or |1>) corresponding to the output of
    Shor’s quantum circuit.'''
    t = fourier(n)

    input = qg.power(qc.ket0, n + n)

    small_h_chunk = qg.power(qc.h, n)
    large_h_chunk = qg.tensor(small_h_chunk, qg.power(qc.i, n))
    first_circuit = numpy.matmul(f, large_h_chunk)

    first_output = qg.application(first_circuit, input)

    leftover_measurement = first_output
    measurement = []
    while len(measurement) < n:
        leftover_measurement, last_measurement = qm.last(leftover_measurement)
        measurement.append(last_measurement)
    
    second_output = qg.application(t, leftover_measurement)

    leftover_measurement = second_output
    measurement = []
    while len(measurement) < n:
        first_measurement, leftover_measurement = qm.first(leftover_measurement)
        measurement.append(first_measurement)
    
    return measurement

def grover(n, k, f):
    '''Assumes n >= 1, k >= 1. Assumes that k is small compared to 2^n.
    Implements the Grover core subroutine. The F parameter is an (n + 1)-qbit
    gate representing a function f : {0, 1}^n -> {0, 1} such that
    SUM_alpha f(alpha) = k. Returns a list or tuple of n classical one-qbit
    states (either |0> or |1>), such that the corresponding n-bit string delta
    usually satisfies f(delta) = 1.'''
    t = math.asin(math.sqrt(k) * (2 ** ((-n) / 2) ))
    l = round((math.pi / (4 * t)) - 0.5)

    h_chunk = qg.power(qc.h, n + 1)
    ketRho = qg.power(qc.ketPlus, n)
    braRho = numpy.conj(ketRho).T

    r = 2 * numpy.matmul(ketRho, braRho) - qg.power(qc.i, n)
    r_chunk = qg.tensor(r, qc.i)

    fr_chunk = numpy.matmul(r_chunk, f)
    
    rotation = fr_chunk
    for _ in range(l - 1):
        rotation = numpy.matmul(rotation, fr_chunk)

    circuit = numpy.matmul(rotation, h_chunk)

    input = qg.tensor(qg.power(qc.ket0, n), qc.ket1)
    output = qg.application(circuit, input)

    #measure first n qbits

    leftover_measurement = output
    measurement = []
    while len(measurement) < n:
        first = qm.first(leftover_measurement)
        first_measurement = first[0]
        leftover_measurement = first[1]
        measurement.append(first_measurement)
    
    return measurement


### DEFINING SOME TESTS ###

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(x):
        return (1 - x[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(x):
        return x
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(x):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(x):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))

def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
            print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print(" delta = " + str(delta))

def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it’s a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    
    #initialize big gamma so we can concatenate
    big_gamma = []
    prediction = None
    while len(big_gamma) < n - 1:
        small_gamma = qu.quantumListToClassicTuple(simon(n, gate))
       
        #concatenate result to big gamma
        big_gamma.append(small_gamma)

        #row reduce:
        big_gamma = qb.reduction(big_gamma)
        big_gamma = qu.removeZeroRow(big_gamma)

    prediction = qu.missingLeadingRow(big_gamma)
    
    for row in big_gamma:
        first_one_index = row.index(1)
        dot_product = qb.dot(prediction, row)

        row_to_add = [0] * n
        row_to_add[first_one_index] = dot_product

        prediction = qb.addition(prediction, row_to_add)

    if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        print(" delta = " + str(delta))
        print(" prediction = " + str(prediction))

def fourierTest(n):
    if n == 1:
        # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print(" got T = ...")
            print(t)
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0, j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        print("passed fourierTest first part")
        # Check that T is unitary.
        tStar = numpy.conj(numpy.transpose(t))
        tStarT = numpy.matmul(tStar, t)
        id = numpy.identity(2**n, dtype=qc.one.dtype)
        if qu.equal(tStarT, id, 0.000001):
            print("passed fourierTest second part")
        else:
            print("failed fourierTest second part")
            print(" T^* T = ...")
            print(tStarT)
            
def shorTest(n, m):
    '''
    Assumes n >= 4, and 2**n >= m**2
    1. Chooses a random k that is coprime to m (math.gcd should help).
    2. Builds the function f that computes powers of k modulo m (qu.powerMod should help).
    3. Runs Shor’s quantum core subroutine on the corresponding gate F.
    4. Interprets the output as an integer b ∈ {0, . . . , 2
    n − 1}.
    5. Prints b. (Later we will improve this step, to make shorTest a real test.)'''
    assert n >= 4 and m != 1
    assert 2 ** n >= m ** 2

    coprimes = [i for i in range(1, m) if math.gcd(m, i) == 1]
    
    k = random.choice(coprimes)

    def f(l):
        l_int = qb.integer(l)
        return qb.string(n, qu.powerMod(k, l_int, m))

    ff = qg.function(n, n, f)

    #brute force to find p
    m_bits = math.ceil(math.log(m, 2))
    p = -1
    for i in range(1, m):
        if qb.integer(f(qb.string(m_bits, i))) == 1:
            p = i
            break

    ff = qg.function(n, n, f)

    shor_p = None
    while shor_p is None:
        #calculate all the steps at once just for cleanness, not most efficient but asymptotically doesn't make a difference
        b = shor(n, ff)
        b_int = qb.integer(qu.quantumListToClassicTuple(b))
        x0 = b_int / (2 ** n)
        _, d = qu.continuedFraction(n, m, x0)

        b = shor(n, ff)
        b_int = qb.integer(qu.quantumListToClassicTuple(b))
        x0 = b_int / (2 ** n)
        _, d_prime = qu.continuedFraction(n, m, x0)

        d_lcm = math.lcm(d, d_prime)
        if qu.powerMod(k, d, m) == 1:
            shor_p = d
        elif qu.powerMod(k, d_prime, m) == 1:
            shor_p = d_prime
        elif qu.powerMod(k, d_lcm, m) == 1:
            shor_p = d_lcm
    
    if p == shor_p:
        print("Passed shorTest")
    else:
        print("failed shorTest")
        print(f"  Actual period: {p}")
        print(f"  Predicted period: {shor_p}")

def groverTest(n, k):
    # Pick k distinct deltas uniformly randomly.
    deltas = []
    while len(deltas) < k:
        delta = qb.string(n, random.randrange(0, 2**n))
        if not delta in deltas:
            deltas.append(delta)
    # Prepare the F gate.
    def f(alpha):
        for delta in deltas:
            if alpha == delta:
                return (1,)
        return (0,)
    fGate = qg.function(n, 1, f)
    # Run Grover’s algorithm up to 10 times.
    qbits = grover(n, k, fGate)
    bits = tuple(map(qu.bitValue, qbits))
    j = 1
    while (not bits in deltas) and (j < 10):
        qbits = grover(n, k, fGate)
        bits = tuple(map(qu.bitValue, qbits))
        j += 1
    if bits in deltas:
        print("passed groverTest in " + str(j) + " tries")
    else:
        print("failed groverTest")
        print(" exceeded 10 tries")
        print(" prediction = " + str(bits))
        print(" deltas = " + str(deltas))

### RUNNING THE TESTS ###

def main():
    fourierTest(3)

if __name__ == "__main__":
    main()


