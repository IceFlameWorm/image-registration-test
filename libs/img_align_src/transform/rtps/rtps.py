import numpy as np
# from scipy import linalg
from . import linalg
from scipy.ndimage import map_coordinates
from .utils import unique
import cv2
cv = cv2


def robust_tps(im1, im2, H, X1_ok, X2_ok, lambd_coef = 0.0001):
    '''
    Args:
        X1_ok and X2_ok: 3 x n, homogenous coordinates
    '''
    imsize1 = im1.shape[:2]
    imsize2 = im2.shape[:2]
    
    lambd = lambd_coef * imsize1[0]*imsize1[1] 

    ## get rid of the coincided points
    ok_nd1 = np.full(X1_ok.shape[1], False)
    _, idx1 = unique(np.round(X1_ok))
    ok_nd1[idx1] = True
    ok_nd2 = np.full(X2_ok.shape[1], False)
    _, idx2 = unique(np.round(X2_ok))
    ok_nd2[idx2] = True
    ok_nd = ok_nd1 & ok_nd2
    X1_nd = X1_ok[:, ok_nd]
    X2_nd = X2_ok[:, ok_nd]

    ## form the tps linear system ##
    x1 = X1_nd[0,:]
    y1 = X1_nd[1,:]
    x2 = X2_nd[0,:]
    y2 = X2_nd[1,:]

    ## get difference between homography est and 
    # real positions
    z1_ = H[2,0]*x1 + H[2,1]*y1 + H[2,2]
    x1_ = (H[0,0]*x1 + H[0,1]*y1 + H[0,2]) / z1_
    y1_ = (H[1,0]*x1 + H[1,1]*y1 + H[1,2]) / z1_
    # x2 = x1_
    # y2 = y1_
    gxn = x1_ - x2
    hyn = y1_ - y2

    n = len(x1_)
    xx = np.repeat(x1_, n).reshape(n,n).T
    yy = np.repeat(y1_, n).reshape(n,n).T
    dist2 = (xx - xx.T)**2 + (yy - yy.T)**2
    dist2.ravel()[::dist2.shape[1]+1] = 1
    K = 0.5 * dist2 * np.log(dist2)
    K.ravel()[::dist2.shape[1]+1] = lambd * 8*np.pi
    K_ = np.zeros((n+3, n+3))
    K_[0:n,0:n] = K
    K_[n,  0:n] = x1_
    K_[n+1,0:n] = y1_
    K_[n+2,0:n] = np.ones(n)
    K_[0:n,  n] = x1_
    K_[0:n,n+1] = y1_
    K_[0:n,n+2] = np.ones(n)
    G_ = np.zeros((n+3,2))
    G_[0:n,0] = gxn
    G_[0:n,1] = hyn
    ########

    ## solve the linear system ##
    W_ = linalg.solve(K_, G_)
    wx = W_[0:n,0]
    wy = W_[0:n,1]
    a = W_[n:n+3,0]
    b = W_[n:n+3,1]

    ## remove outliers based on the distribution of weights
    outlier = (abs(wx)>3*np.std(wx)) | (abs(wy)>3*np.std(wy))

    inlier_idx = np.arange(len(x1_))
    for kiter in range(10):
        if sum(outlier) < 0.0027*n:
            break
        ok = ~outlier
        inlier_idx = inlier_idx[ok]
        K_ = K_[np.concatenate((ok, [True,True,True])),:][:,np.concatenate((ok, [True,True,True]))]
        G_ = G_[np.concatenate((ok, [True,True,True])),:]
        W_ = linalg.solve(K_, G_)
        n = len(inlier_idx)
        wx = W_[0:n,0]
        wy = W_[0:n,1]
        a = W_[n:n+3,0]
        b = W_[n:n+3,1]
        outlier = (abs(wx)>3*np.std(wx)) | (abs(wy)>3*np.std(wy))

    ok = np.full(len(x1), False)
    ok[inlier_idx] = True
    x1 = x1[ok]
    y1 = y1[ok]
    x2 = x2[ok]
    y2 = y2[ok]
    x1_ = x1_[ok]
    y1_ = y1_[ok]
    #gxn: dist between moved x and x in im2
    #hyn: dist between moved y and y in im2
    gxn = gxn[ok]
    hyn = hyn[ok]
    ########

    res_X1 = np.vstack([x1, y1])
    res_X2 = np.vstack([x2, y2])
    res_X1_ = np.vstack([x1_, y1_])
    res_gh = np.vstack([gxn, hyn])
    res_weighs = {
        'wx': wx,
        'wy': wy,
        'a': a,
        'b': b
    }
    
    return res_X1, res_X2, res_X1_, res_gh, res_weighs