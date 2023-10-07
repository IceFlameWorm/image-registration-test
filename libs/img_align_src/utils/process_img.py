import cv2
import numpy as np
from .mesh import grid2mesh
from skimage.filters import threshold_sauvola

def otsu_thresh(im):
    to_binarize = np.copy(np.uint8(im))
    ret2, th2 = cv2.threshold(to_binarize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2

def adaptive_threshold(im):
    return 255 - cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 4)
def savola(img):
    thresh_sauvola = threshold_sauvola(img, window_size=51)
    binary_sauvola = img > thresh_sauvola
    ret_sauvola = binary_sauvola.astype(np.uint8) * 255
    return ret_sauvola

def warp_img(src_img_bgr, src_meshgrid_xy, tgt_meshgrid_xy):
    src_meshgrid_x, src_meshgrid_y = src_meshgrid_xy
    tgt_meshgrid_x, tgt_meshgrid_y = tgt_meshgrid_xy

    mesh_rsize, mesh_csize = tgt_meshgrid_x.shape
    grid_rsize, grid_csize = mesh_rsize - 1, mesh_csize - 1
    
    tgt_br_pt_x, tgt_br_pt_y = tgt_meshgrid_x[-1, -1], tgt_meshgrid_y[-1, -1]
    tgt_img_w, tgt_img_h = int(tgt_br_pt_x) + 1, int(tgt_br_pt_y) + 1
    tgt_img_bgr = np.zeros([tgt_img_h, tgt_img_w, 3], dtype= np.uint8)

    for gridx in range(grid_rsize):
        for gcidx in range(grid_csize):
            _, src_vertex_pts = grid2mesh(gridx, gcidx, src_meshgrid_x, src_meshgrid_y)
            src_vertex_pts = np.array(src_vertex_pts, dtype = np.float32)
            _, tgt_vertex_pts = grid2mesh(gridx, gcidx, tgt_meshgrid_x, tgt_meshgrid_y)
            tgt_tl_pt, tgt_br_pt = tgt_vertex_pts[0], tgt_vertex_pts[2]
            tgt_min_x, tgt_min_y = tgt_tl_pt
            tgt_min_x, tgt_min_y = int(tgt_min_x), int(tgt_min_y)
            tgt_max_x, tgt_max_y = tgt_br_pt
            tgt_max_x, tgt_max_y = int(tgt_max_x), int(tgt_max_y)
            tgt_w, tgt_h = tgt_max_x - tgt_min_x, tgt_max_y - tgt_min_y
            tgt_vertex_pts = np.array([[0,0], [tgt_w, 0], [tgt_w, tgt_h], [0, tgt_h]], dtype = np.float32)
            local_H = cv2.getPerspectiveTransform(src_vertex_pts, tgt_vertex_pts)
            local_tgt_bgr = cv2.warpPerspective(src_img_bgr, local_H, (tgt_w, tgt_h))
            tgt_img_bgr[tgt_min_y: tgt_min_y + tgt_h,
                        tgt_min_x: tgt_min_x + tgt_w, :] = local_tgt_bgr

    return tgt_img_bgr


def uniform_blend(img1, img2):
    # grayscale
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)
    result = (img1.astype(np.float64) + img2.astype(np.float64))

    g1, g2 = gray1 > 0, gray2 > 0
    g = g1 & g2
    mask = np.expand_dims(g * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1
    result *= mask
    result = result.astype(np.uint8)

    return result

def get_padded_boundary(dewarped_gray):
    """Some times the sample image is not complete, 
       In this case there will be padding when apply
       the dewarping transform. This function aims to
       output a map that points out all paddings. 

    Args:
        dewarped_gray (np.ndarray): uint8 array presenting
                         the dewarped image

    Returns:
        np.ndarray: a map that has same shape as input image.
                    it's values are 1 and 0 where 1 means the 
                    pixel was padded when applying warping
    """
    not_bk_map = (dewarped_gray > 0).astype(np.float32)
    h, w = dewarped_gray.shape
    left_map = np.cumsum(not_bk_map, axis=1) > 0
    right_map = np.cumsum(not_bk_map[:,::-1], axis=1) > 0

    left_inds = np.argmax(left_map, axis = 1)
    left_inds[left_inds == 0] = left_inds[left_inds == 0] + \
                                (1 - left_map[:,0][left_inds == 0])*w
    
    right_inds = w - np.argmax(right_map, axis=1)
    bk_pad = np.ones_like(dewarped_gray, dtype=np.uint8)
    for i, (left_ind, right_ind) in enumerate(zip(left_inds, right_inds)):
        bk_pad[i, left_ind:right_ind] = 0
        # print(left_ind, right_ind)
    # cv2.imwrite('debug.png', bk_pad*255)
    return bk_pad


def get_image_score(img_bgr_1, img_bgr_2):
    """get a score for matching quaily given the
       dewarped image and template

    Args:
        img_bgr_1 (np.ndarray): dewarped image
        img_bgr_2 (np.ndarray): template image

    Returns:
        score: how good is the matching result
    """
    dewarped_gray = cv2.cvtColor(img_bgr_1, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(img_bgr_2, cv2.COLOR_BGR2GRAY)
    # print('sample shape: ', dewarped_gray.shape)
    # print('template shape: ', template_gray.shape)
    img_1_shape = dewarped_gray.shape[:2]
    img_2_shape = template_gray.shape[:2]
    if img_1_shape != img_2_shape:
        dewarped_gray = cv2.resize(dewarped_gray, img_2_shape[::-1])
    dewarpe_pad = get_padded_boundary(dewarped_gray)
    # cv2.imwrite('dewarped_orig.png', dewarped_gray)
    dewarped_bin = adaptive_threshold(dewarped_gray)
    template_bin = otsu_thresh(template_gray)
    kernel = np.ones((5, 5), np.uint8)
    dewarped_eroded = cv2.erode(dewarped_bin, kernel, iterations=1)

    dewarped_fore = (dewarped_eroded == 0).astype(np.uint8)
    # cv2.imwrite('dewarped_fore.png', (dewarped_fore*255).astype(np.uint8))
    template_fore = (template_bin == 0).astype(np.uint8)
    # cv2.imwrite('template_fore.png', (template_fore*255).astype(np.uint8))
    template_fore[dewarpe_pad == 1] = 0
    # cv2.imwrite('dewarpe_pad.png', (dewarpe_pad*255).astype(np.uint8))
    n_pixel = np.sum(template_fore)
    template_fore[dewarped_fore == 1] = 0
    # cv2.imwrite('overlapped.png', (template_fore*255).astype(np.uint8))
    missed_pixel = np.sum(template_fore)
    score = (n_pixel - missed_pixel) / n_pixel
    # print('matching score: ', score)
    return score

def blend_imgs(img_bgr_1, img_bgr_2):
    img_1_shape = img_bgr_1.shape[:2]
    img_2_shape = img_bgr_2.shape[:2]
    if img_1_shape != img_2_shape:
        img_bgr_1 = cv2.resize(img_bgr_1, img_2_shape[::-1])

    # blended_img_bgr = uniform_blend(img_bgr_1, img_bgr_2)
    h, w = img_bgr_1.shape[:2]
    blended_img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    blended_img_bgr[:,:,1] = img_bgr_1[:,:,1]
    blended_img_bgr[:,:,2] = img_bgr_2[:,:,2]
    return blended_img_bgr