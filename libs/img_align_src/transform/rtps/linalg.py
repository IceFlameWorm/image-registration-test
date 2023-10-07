import numpy as np


def solve(A, Y, solver = 'torch'):
    X = globals()[f'_{solver}_solve'](A, Y)
    return X


def _torch_solve(A, Y):
    import torch
    X = torch.linalg.solve(
        torch.from_numpy(A.astype(np.float32)), 
        torch.from_numpy(Y.astype(np.float32))
    ).numpy()

    return X


def _scipy_solve(A, Y):
    import scipy
    X = scipy.linalg.solve(A, Y)
    return X
