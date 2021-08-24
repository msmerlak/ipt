import torch

def eig(M, V0 = None, max_iter = 500, eigenvalues = True):

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

      err = 1
      it = 0

      while err > 2*torch.finfo(M.dtype).eps and it < max_iter:
          # On the GPU the error condition is the time bottleneck

          it += 1
          W = F(V, id, theta, delta)
          err = torch.max(torch.abs(W-V))
          V = W

      D = torch.diag(d + torch.diag(torch.matmul(delta, V)))

      print('Residual DPT:', torch.linalg.norm(M @ V - V @ D).item())

      if eigenvalues:
          return(D, V)
      else:
          return(V)
