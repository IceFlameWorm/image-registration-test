import cv2
import numpy as np
from .homography import HM_ransac, check_H, affine_ransac
from .rtps import robust_tps
from ...utils.process_img import warp_img, blend_imgs, get_image_score  
# from scipy import linalg
from . import linalg


class RTPSTransformer(object):
    default_config = {
        'lambd_coef': 0.0002,
        'Homography': 'orig' # orig or cv2
    }

    def __init__(self, template_img_bgr, sample_img_bgr,
                 template_mkpts, sample_mkpts,
                 config = {}):
        self._template_img_bgr = template_img_bgr
        self._sample_img_bgr = sample_img_bgr
        self._template_mkpts = template_mkpts
        self._sample_mkpts = sample_mkpts
        self._config = {**self.default_config, **config}

    def estimate(self):
        ## Estimate the global homography
        self._estimate_global_H()

        ## Get the overlap
        self._get_overlap()
        ## Do robust tps
        self._robust_tps()

    def _estimate_global_H(self):
        homography = self._config["Homography"]
        if homography == 'orig':
            self._estimate_global_H_orig()
        elif homography == 'cv2':
            self._estimate_global_H_cv2()
        else:
            raise Exception(f"unsupported homographpy config: {homography}")

    def _estimate_global_H_orig(self):
        X1, X2 = self._template_mkpts, self._sample_mkpts
        
        X1 = np.append(X1, np.ones([X1.shape[0], 1]), axis = 1)
        X2 = np.append(X2, np.ones([X2.shape[0], 1]), axis = 1)
        X1, X2 = X1.T, X2.T
        H, ok, score = HM_ransac(X1, X2, 500, 0.1)
        if not check_H(H):
            template_w =  self._template_img_bgr.shape[1]
            H, ok, score = affine_ransac(X1, X2, 500, template_w)
        X1_ok = X1[:, ok]
        X2_ok = X2[:, ok]
        self._global_H = H
        self._template_mkpts_ok = X1_ok
        self._sample_mkpts_ok = X2_ok

    def _estimate_global_H_cv2(self):
        X1, X2 = self._template_mkpts, self._sample_mkpts # N x 2
        H, mask = cv2.findHomography(X1, X2, method=cv2.RANSAC)
        X1, X2 = X1.T, X2.T # 2 x N
        X1 = np.append(X1, np.ones([X1.shape[0], 1]), axis = 1) # 3 x N
        X2 = np.append(X2, np.ones([X2.shape[0], 1]), axis = 1) # 3 x N
        X1_ok, X2_ok = X1, X2
        self._global_H = H
        self._template_mkpts_ok = X1_ok
        self._sample_mkpts_ok = X2_ok

    def _get_overlap(self):
        im1 = self._template_img_bgr
        im2 = self._sample_img_bgr
        H = self._global_H

        imsize1 = im1.shape[:2]
        imsize2 = im2.shape[:2]

        ## 4 corner points of im1
        box1 = np.array([[0, im1.shape[1]-1, im1.shape[1]-1, 0],
                         [0, 0,              im1.shape[0]-1, im1.shape[0]-1],
                         [1, 1,              1,              1]])

        ## 4 corner points of im2
        box2 = np.array([[0, im2.shape[1]-1, im2.shape[1]-1, 0],
                         [0, 0,              im2.shape[0]-1, im2.shape[0]-1],
                         [1, 1,              1,              1]])

        ## the width and height of the result image.
        mosaich, mosaicw = imsize1
        u0, u1 = 0, mosaicw - 1
        v0, v1 = 0, mosaich - 1

        ## the overlapping area of im2 in the result image
        box2_ = linalg.solve(H, box2)
        box2_[0,:] = box2_[0,:] / box2_[2,:]
        box2_[1,:] = box2_[1,:] / box2_[2,:]

        margin = 0.2 * min(imsize1[0],imsize1[1])
        u0_im_ = max(min(box2_[0,:]) - margin, u0)
        u1_im_ = min(max(box2_[0,:]) + margin, u1)
        v0_im_ = max(min(box2_[1,:]) - margin, v0)
        v1_im_ = min(max(box2_[1,:]) + margin, v1)
        offset_u0_ = int(np.ceil(u0_im_ - u0))
        offset_u1_ = int(np.floor(u1_im_ - u0))
        offset_v0_ = int(np.ceil(v0_im_ - v0))
        offset_v1_ = int(np.floor(v1_im_ - v0))
        imw_ = int(np.floor(offset_u1_ - offset_u0_ + 1))
        imh_ = int(np.floor(offset_v1_ - offset_v0_ + 1))

        ## the overlapping region of im1 in the image coordiantes of image 2
        box1_2 = np.dot(H, box1)
        box1_2[0,:] = box1_2[0,:] / box1_2[2,:]
        box1_2[1,:] = box1_2[1,:] / box1_2[2,:]

        sub_u0_ = max([0, min(box1_2[0,:])])
        sub_u1_ = min([imsize2[1]-1, max(box1_2[0,:])])
        # sub_v0_ = max([0, min(box1_2[1,:])]) - margin # may be a mistake
        sub_v0_ = max([0, min(box1_2[1,:])])
        sub_v1_ = min([imsize2[0]-1, max(box1_2[1,:])])

        self._overlap = {
            'offset_u0_': offset_u0_,
            'offset_u1_': offset_u1_,
            'offset_v0_': offset_v0_,
            'offset_v1_': offset_v1_,
            'imw_': imw_,
            'imh_': imh_,
            'sub_u0_': sub_u0_,
            'sub_u1_': sub_u1_,
            'sub_v0_': sub_v0_,
            'sub_v1_': sub_v1_
        }

    def _robust_tps(self):
        im1, im2 = self._template_img_bgr, self._sample_img_bgr
        H = self._global_H
        X1_ok, X2_ok = self._template_mkpts_ok, self._sample_mkpts_ok
        lambd_coef = self._config['lambd_coef']
        res_X1, res_X2, res_X1_, res_gh, res_weights = robust_tps(im1, im2, H, X1_ok, X2_ok, lambd_coef)
        self._template_mkpts_tps = res_X1
        self._sample_mkpts_tps = res_X2
        self._template_mkpts_tps_ = res_X1_
        self._parallex_ = res_gh
        self._tps_weights = res_weights

    @staticmethod
    def transform_template_pt(template_pt, H, tps_weights, template_mkpts_tps_,
                              parallex, overlap, K_smooth = 5):
        '''
        Args:
            template_pt: ndarray, (2, ),
            template_mkpts_tps_: ndarray, (2, n),
            parallex: ndarray, (2, n)
        '''
        ## Do global homographic transform
        template_hpt = np.append(template_pt, 1) # (3, )
        # import pdb
        # pdb.set_trace()
        template_hpt_ = H @ template_hpt # (3, )
        template_pt_ = template_hpt_ / (template_hpt_[2] + 1e-8)
        u_, v_ = template_pt_[:2]

        ## Do tps transformation
        gx, hy = 0, 0
        offset_u0_, offset_u1_ = overlap['offset_u0_'], overlap['offset_u1_']
        offset_v0_, offset_v1_ = overlap['offset_v0_'], overlap['offset_v1_']
        if (offset_u0_ <= u_ <= offset_u1_) and (offset_v0_ <= v_ <= offset_v1_):
            x1_, y1_ = template_mkpts_tps_
            n = len(x1_)
            gx_sub, hy_sub = 0, 0
            wx, wy = tps_weights['wx'], tps_weights['wy']
            a, b = tps_weights['a'], tps_weights['b']
            for kf in range(n):
                dist2 = (u_ - x1_[kf]) ** 2 + (v_ - y1_[kf]) ** 2
                rbf = 0.5 * dist2 * np.log(dist2)
                gx_sub += wx[kf] * rbf
                hy_sub += wy[kf] * rbf

            gx_sub += a[0] * u_ + a[1] * v_ + a[2]
            hy_sub += b[0] * u_ + b[1] * v_ + b[2]
            gx, hy = gx_sub, hy_sub

        ## smooth transition to global transform
        gxn, hyn = parallex
        eta_d0 = 0
        eta_d1 = K_smooth * max(abs(np.concatenate([gxn, hyn])))
        sub_u0_, sub_u1_ = overlap['sub_u0_'], overlap['sub_u1_']
        sub_v0_, sub_v1_ = overlap['sub_v0_'], overlap['sub_v1_']
        sub_u0_ = sub_u0_ + min(gxn)
        sub_u1_ = sub_u1_ + max(gxn)
        sub_v0_ = sub_v0_ + min(hyn)
        sub_v1_ = sub_v1_ + max(hyn)
        dist_horizontal = np.maximum(sub_u0_-u_, u_-sub_u1_)
        dist_vertical = np.maximum(sub_v0_-v_, v_-sub_v1_)
        dist_sub = np.maximum(dist_horizontal, dist_vertical)
        dist_sub = np.maximum(0, dist_sub)
        eta = (eta_d1 - dist_sub) / (eta_d1 - eta_d0)
        # eta[dist_sub < eta_d0] = 1
        # eta[dist_sub > eta_d1] = 0
        if dist_sub < eta_d0:
            eta = 1
        elif dist_sub > eta_d1:
            eta = 0

        gx *= eta
        hy *= eta

        res_u_ = u_ - gx
        res_v_ = v_ - hy
        return np.array([res_u_, res_v_])

    @staticmethod
    def transform_template_pts(template_pts, H, tps_weights, templates_mkpts_tps_,
                               parallex, overlap, K_smooth = 5):
        res_pts = [
            RTPSTransformer.transform_template_pt(template_pt, H,
                tps_weights, templates_mkpts_tps_, parallex, overlap, K_smooth)
            for template_pt in template_pts
        ]
        return res_pts

    @staticmethod
    def transform_template_meshgrid(template_meshgrid_xy, H, tps_weights, template_mkpts_tps_,
                                    parallex, overlap, K_smooth = 5):
        template_meshgrid_x, template_meshgrid_y = template_meshgrid_xy
        sample_meshgrid_x = np.zeros_like(template_meshgrid_x)
        sample_meshgrid_y = np.zeros_like(template_meshgrid_y)

        mesh_rsize, mesh_csize = template_meshgrid_x.shape

        for mridx in range(mesh_rsize):
            for mcidx in range(mesh_csize):
                template_pt_x = template_meshgrid_x[mridx, mcidx]
                template_pt_y = template_meshgrid_y[mridx, mcidx]
                sample_pt_x, sample_pt_y = RTPSTransformer.transform_template_pt(
                    np.array([template_pt_x, template_pt_y]),
                    H, tps_weights, template_mkpts_tps_,
                    parallex, overlap, K_smooth
                )
                sample_meshgrid_x[mridx, mcidx] = sample_pt_x
                sample_meshgrid_y[mridx, mcidx] = sample_pt_y

        return sample_meshgrid_x, sample_meshgrid_y

    @staticmethod
    def transform_img(src_img_bgr, src_meshgrid_xy, tgt_meshgrid_xy):
        return warp_img(src_img_bgr, src_meshgrid_xy, tgt_meshgrid_xy)

    @staticmethod
    def blend_imgs(img_bgr_1, img_bgr_2):
        return get_image_score(img_bgr_1, img_bgr_2), blend_imgs(img_bgr_1, img_bgr_2)