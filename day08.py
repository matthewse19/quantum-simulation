import qUtilities as qu
import qConstants as qc
import qGates as qg
import numpy

top = numpy.matmul(qg.tensor(qc.h, qc.h), numpy.matmul(qc.cnot, qg.tensor(qc.h, qc.h)))
top2 = numpy.matmul(qp)
bottom = numpy.matmul(qc.swap, numpy.matmul(qc.cnot, qc.swap))
print(top)
print(bottom)