import numpy
from numpy import linalg
from numpy import matrix
import math

# Let G be an m times n genotype matrix whose rows represent SNPs and columns represent individuals.
# We assume G is a numpy matrix consisting of elements 0, 1 and 2.
# Then whiten(G) returns an m times (n + 1) pseudogenotype matrix F of the form F = G*X for some matrix X.
# The pseudogenotype matrix has the property that its GRM is as close to the identity as possible, namely (m/n)I - (m/(n(n + 1)))11^T.
# The pseudogenotype matrix F already have row means zero, and the row standard deviations can be scaled to one with row_normalize().
# If the algorithm seems to take forever you might want to switch the autostop to False and experiment with fixed numbers of iterations.
def whiten(G, autostop = True, tol = 0.00001, iterations = 100):
    G = G.astype("float") # Making sure the input is interpreted as a float matrix.
    m = G.shape[0] # The number of rows.
    n = G.shape[1] # The number of columns.
    D = numpy.ones(m) # Here we save the scalars we use during the iterations.
    if autostop is True:
        while True:
            E = numpy.copy(D) # Stupid python.
            for i in range(m): # Scaling the (Euclidean) length of each row vector to one.
                D[i] = D[i]*linalg.norm(G[i])
                G[i] = G[i]/linalg.norm(G[i])
            G = linalg.qr(G)[0] # Orthogonalizing the columns.
            G = math.sqrt(m/n)*G # Scaling the whole thing so that it's the rows and not the columns that can have unit (Euclidean) length.
            if linalg.norm(D - E) < tol:
                break
    else:
        for k in range(iterations):
            for i in range(m): # Scaling the (Euclidean) length of each row vector to one.
                D[i] = D[i]*linalg.norm(G[i])
                G[i] = G[i]/linalg.norm(G[i])
            G = linalg.qr(G)[0] # Orthogonalizing the columns.
            G = math.sqrt(m/n)*G # Scaling the whole thing so that it's the rows and not the columns that can have unit (Euclidean) length.
    a = (1 - math.sqrt(1/(n + 1)))/n # A little trick to make the row means zero.
    L = numpy.identity(n) - a*numpy.ones((n, n))
    G = G*L
    for i in range(m): # Multiplying the rows of G by the elements of D, using numpy.diag(D)*G is rather heavy.
        G[i] = D[i]*G[i]
    G = numpy.hstack([G, - numpy.sum(G, axis = 1)])
    return(G)

# The rest is for checking.

# Scales the (Euclidean) length of each row vector of a matrix to one.
def row_normalize(G):
    G = G.astype("float") # Making sure the input is interpreted as a float matrix.
    for i in range(G.shape[0]):
        G[i] = G[i]/linalg.norm(G[i])
    return(G)

# Checks whether the column space of F is contained in the column space of G.
def same_column_space(F, G, tol = 0.00001):
    answer = True
    for j in range(F.shape[1]):
        if linalg.lstsq(G, F.T[j].T)[1] > tol:
            answer = False
    return(answer)