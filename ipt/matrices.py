import scipy.sparse

def symmetrize(M):
    return (M + M.T)/2

def almost_degenerate(size, density, epsilon, symmetric = False):
    N = size
    D = scipy.sparse.diags([range(N)], [0], shape = (N,N))
    M = D + epsilon*scipy.sparse.rand(N, N, density=density)
    if symmetric:
        H = symmetrize(M)
    else:
        H = M

    if density >= 0.5:
        H = H.toarray()
    else:
        H.tocsr()

    return(H)

def perturbative_matrix(size, density, epsilon, symmetric = False):
    N = size
    D = scipy.sparse.diags([range(1, N+1)], [0], shape = (N,N))
    M = D + epsilon*scipy.sparse.rand(N, N, density=density)
    if symmetric:
        H = symmetrize(M)
    else:
        H = M

    if density >= 0.5:
        H = H.toarray()
    else:
        H.tocsr()

    return(H)
