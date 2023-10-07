import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def cell(m, n=None):
    a = []
    for i in range(m):
        a.append([])
        if n is None:
            for j in range(m):
                a[i].append(None)
        else:
            for j in range(n):
                a[i].append(None)
    return a

def unique(A):
    ar, idx = np.unique(A, return_index=True, axis=1)
    return ar, idx
