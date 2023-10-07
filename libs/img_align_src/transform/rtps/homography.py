import numpy as np
import random
from scipy import linalg
from .utils import cell
import cv2
random.seed(0)

def HM_ransac(X1, X2, Nr, min_dis):
    """get homograph matr with ransac

    Args:
        X1 (np.ndarray): 3*n_points array for all
                         matched points in im1
        X2 (np.ndarray): 3*n_points array for all
                         matched points in im2
        Nr (int): number of rounds
        min_dis (int): ransac threshold 

    Returns:
        H (np.ndarray): homograph matrix
        ok (np.ndarray): numpy array for if the point is ok 
        score (int): max number of points matched 
    """
    N = X1.shape[1]

    u = X1[0,:].reshape(-1,1)
    v = X1[1,:].reshape(-1,1)
    u_ = X2[0,:].reshape(-1,1)
    v_ = X2[1,:].reshape(-1,1)

    scale = 1 / np.mean(np.vstack((u, u_, v, v_)))
    u = u * scale
    v = v * scale
    u_ = u_ * scale
    v_ = v_ * scale

    A1 = np.hstack([np.zeros((N,3)),      -u, -v, -np.ones((N,1)), v_*u, v_*v, v_])
    A2 = np.hstack([u, v, np.ones((N,1)), np.zeros((N,3)),         -u_*u, -u_*v, -u_])
    # print(np.vstack((A1, A2)))
    if min_dis > 0:
        H = cell(Nr, 1); ok = cell(Nr, 1); score = np.zeros(Nr, 'int')
        A = cell(Nr, 1)
        for t in range(Nr):
            subset = random.sample(list(range(N)), 4)
            A[t] = np.vstack((A1[subset,:], A2[subset,:]))
            U,S,V = linalg.svd(A[t])
            h = V.T[:,8]
            H[t] = h.reshape(3,3)
            # print(H[t])
            dis2 = np.dot(A1, h)**2 + np.dot(A2, h)**2
            # print('dis2: ', dis2)
            ok[t] = dis2 < min_dis * min_dis
            score[t] = sum(ok[t])
        # import pdb
        # pdb.set_trace()
        score, best = max(score), np.argmax(score)
        ok = ok[best]
        A = np.vstack((A1[ok,:], A2[ok,:]))
        U,S,V = linalg.svd(A, 0)
        h = V.T[:,8]
        H = h.reshape(3,3)

    else:
        A = np.vstack((A1, A2))
        U,S,V = linalg.svd(A, 0)
        h = V.T[:,8]
        H = h.reshape(3,3)
    
    H = np.dot(np.dot(np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]]), H),
               np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))
    return H, ok, score

def affine_ransac(X1, X2, Nr, template_w):
    """get affine matr with "ransac"

    Args:
        X1 (np.ndarray): 3*n_points array for all
                         matched points in im1
        X2 (np.ndarray): 3*n_points array for all
                         matched points in im2
        Nr (int): number of rounds
        min_dis (int): ransac threshold 

    Returns:
        out_A (np.ndarray): homograph matrix that is in fact an
                            affine transform
        picked_ok (np.ndarray): numpy array for if the point is ok 
        max_score (int): max number of points matched 
    """
    N = X1.shape[1]
    in_pts = X1.T[:,:2].astype(np.float32)
    out_pts = X2.T[:,:2].astype(np.float32)
    # print(np.vstack((A1, A2)))
    thresh = 0.05 * template_w
    
    scores = np.zeros(Nr, 'int')
    ok = cell(Nr, 1)
    max_score = 0
    picked_A = [[1,0,0],[0,1,0]]
    picked_score = 0
    picked_ok = np.ones(N, np.bool8) #just to get all True
    for t in range(Nr):
        subset = random.sample(list(range(N)), 3)
        
        A = cv2.getAffineTransform(in_pts[subset,:], out_pts[subset,:])
        
        # print(H[t])
        dis2 = np.sum((np.dot(A, X1) - X2[:2, :])**2, axis=0)
        # print('dis2: ', dis2)
        ok[t] = dis2 < thresh * thresh
        scores[t] = sum(ok[t])
        
        if scores[t] > max_score:
            picked_A = A
            picked_ok = ok[t]
            max_score = scores[t]

    # import pdb
    # pdb.set_trace()
    out_A = np.zeros((3,3), np.float32)
    out_A[:2, :] = picked_A
    out_A[2,2] = 1.
    return out_A, picked_ok, max_score

def check_H(H):
    """check if H is valid, we assume that the image is not 
       reversed so h00 and h11 must be positive after normalize 
       with z. This check is quite empirical

    Args:
        H (np.ndarray): homography matrix

    Returns:
        bool: if the H is problematic
    """    
    if H[0,0] / H[2,2] < 0:
        return False
    if H[1,1] / H[2,2] < 0:
        return False
    return True

