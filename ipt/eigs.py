import aa # Anderson acceleration from https://github.com/cvxgrp/aa
import numpy as np
import time

def eigs(H, i = 0, v0 = None, mem = 10, maxiter = 1000, tol = 1e-12, quiet = True):

    dim = H.shape[0]

    # Epstein-Nesbet partitioning
    H0 = H.diagonal()

    # Reduced resolvent
    gaps = H0[i] - H0; gaps[i]=1; R0 = 1/gaps; R0[i] = 0

    # Initialize Anderson accelerator
    aa_wrk = aa.AndersonAccelerator(dim, mem)

    # Unperturbed eigenvector (basis state)
    e = np.zeros(dim); e[i] = 1

    # Q_IPT, eq. (3) in main text
    def Q(v, R0, H0):
        H0v = np.multiply(H0, v)
        H1v = H @ v - H0v
        return(e + np.multiply(R0, H1v - H1v[i]*v), H0v, H1v)

    if v0 is not None:
        w = v0
    else:
        w = e

    err = []
    l = 0

    tic = time.time()

    while l <= maxiter:

        v = w

        # Iterate Q
        w, H0v, H1v = Q(v, R0, H0)
        E = H0[i] + H1v[i]

        # Residual error
        err.append(np.linalg.norm(H0v + H1v - E * v)/np.linalg.norm(v))
        if err[-1] < tol:
            break

        if not quiet:
            print('Iterations:',  l)
            print('Residual:',  err[-1])

        # Anderson acceleration
        aa_wrk.apply(w, v)
        l += 1

    toc = time.time()
    timing = toc - tic

    return({'iterations': l, 'time': timing, 'eigenvalue': E, 'eigenvector': v, 'residual' : err[-1]})
