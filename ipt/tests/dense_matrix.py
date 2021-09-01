from matrices import *
import numpy as np

dense = perturbative_matrix(size=2000, density=1, epsilon=.1, symmetric=False)

eig(dense, quiet=False)

tic = time.time()
D, V = np.linalg.eig(dense)
toc = time.time()
res = np.linalg.norm(dense @ V - V @ np.diag(D))
print('\nTime DGEEV:', toc - tic)
print('Residual DGEEV:', res)
