"""
Compute the full set of eigenvectors/eigenvalues of a near-diagonal matrix.
"""

import torch
import time

def eig(M, V0 = None, max_iter = 1000, tol = 1e-12, return_eigenvectors = True, quiet = False):

    if not torch.is_tensor(M):
        M = torch.tensor(M)

    d = M.diagonal()

    def F(V, id, theta, delta):
        n = V.size()[0]
        deltaV = torch.matmul(delta, V)
        W = deltaV - torch.matmul(V, torch.diag(torch.diag(deltaV)))
        return(id - torch.mul(theta, W))


    theta = torch.triu(1/(d.unsqueeze(1) - d), diagonal = 1)
    theta = theta - theta.t()

    delta = M - torch.diag(d)

    id = torch.eye(M.size(0), dtype = M.dtype, device = M.device)

    if V0 == None:
        V = id
    else:
        V = V0

    err = []
    it = 0

    tic = time.time()
    while it < max_iter:

        it += 1
        W = F(V, id, theta, delta)
        err.append(torch.max(torch.abs(W-V))/torch.max(torch.abs(W)))
        V = W

        D = torch.diag(d + torch.diag(torch.matmul(delta, V)))
        if err[-1] < max(tol, 2*torch.finfo(M.dtype).eps):
                    break
    toc = time.time()
    if not quiet:
        print('Time:', toc - tic)
        print('Iterations:', it)
        print('Residual:', torch.linalg.norm(M @ V - V @ D).item())

    if return_eigenvectors:
        return(D, V)
    else:
        return(torch.diag(D))



if __name__ == '__main__':

    from matrices import *

    import numpy as np

    dense = perturbative_matrix(size = 2000, density=1, epsilon=.1, symmetric=False)

    eig(dense, quiet=False);

    tic = time.time()
    D,V = np.linalg.eig(dense);
    toc = time.time()
    res = np.linalg.norm(dense @ V - V @ np.diag(D))
    print('\nTime DGEEV:', toc - tic)
    print('Residual DGEEV:', res)
