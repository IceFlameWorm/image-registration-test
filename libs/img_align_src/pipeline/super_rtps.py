import cv2
import numpy as np
from ..transform.rtps.rtps_transform import RTPSTransformer
from ..utils.mesh import transform_pt
from scipy.spatial.distance import cdist
import math


class SuperRTPS(object):
    def __init__(self, super_dm, rtps_config, template_bgr,
                 sample_bgr, resize = [480, 640], mesh_rsize = 31,
                 mesh_csize = 31):
        self._super_dm = super_dm
        self._rtps_config = rtps_config
        self._template_bgr = template_bgr
        self._sample_bgr = sample_bgr
        self._resize = resize
        self._mesh_rsize = mesh_rsize
        self._mesh_csize = mesh_csize
        self.mlrd = multiline_region_detector() # 多线段检测器

    def align(self):
        ## Scale the images
        (self._template_resized_bgr, self._template_x_scale,self._template_y_scale) = self._resize_img(self._template_bgr, self._resize)
        (self._sample_resized_bgr, self._sample_x_scale,self._sample_y_scale) = self._resize_img(self._sample_bgr, self._resize)
        

        ## detect and match feature points

        self._feature_matches = self._super_dm.detect_match(
            self._template_resized_bgr, self._sample_resized_bgr
        )

        template_mkpts = self._feature_matches['mkpts0']
        sample_mkpts = self._feature_matches['mkpts1']
        mconf = self._feature_matches['mconf']
        # 去掉异点
        # template_mkpts,sample_mkpts = remove_difpt(template_mkpts,sample_mkpts,expand_w=self._resize[0])
        # 去掉孤点
        thres = (self._resize[0] * self._resize[1]) ** 0.5 /4
        template_mkpts,sample_mkpts,mconf = remove_isopt(template_mkpts,sample_mkpts,mconf,thres)
        # 去除多横线区域、置信度低的端点
        if self.mlrd.detect(self._template_resized_bgr):
            template_mkpts,sample_mkpts,mconf = self.mlrd.remove_mlr(template_mkpts,sample_mkpts,mconf,conf_thres=0.8)
        # 修改结果
        self._feature_matches['mkpts0'],self._feature_matches['mkpts1'],self._feature_matches['mconf'] = template_mkpts,sample_mkpts,mconf
        
        ## estimate the transformation model
        self._rtps_trans = RTPSTransformer(self._template_resized_bgr, self._sample_resized_bgr,
            template_mkpts, sample_mkpts, self._rtps_config)

        # print("2"*20)
        # print(self._rtps_trans._template_mkpts.min(), self._rtps_trans._sample_mkpts.min())
        
        self._rtps_trans.estimate()

        ## calculate the template and sample meshgrids
        self._cal_template_meshgrid()
        self._cal_sample_meshgrid()

    def _resize_img(self, img_bgr, resize):
        img_h, img_w = img_bgr.shape[:2]
        resize = tuple(resize)
        resize_w, resize_h = resize
        x_scale = resize_w / img_w
        y_scale = resize_h / img_h
        resized_img_bgr = cv2.resize(img_bgr, resize, interpolation= cv2.INTER_AREA)
        return resized_img_bgr, x_scale, y_scale

    def _cal_template_meshgrid(self):
        tresized_h, tresized_w = self._template_resized_bgr.shape[:2]
        scaled_tmesh_w = np.linspace(0, tresized_w - 1, self._mesh_csize)
        scaled_tmesh_h = np.linspace(0, tresized_h - 1, self._mesh_rsize)
        tmesh_w = [x / self._template_x_scale for x in scaled_tmesh_w]
        tmesh_h = [y / self._template_y_scale for y in scaled_tmesh_h]
        self._template_meshgrid_xy = np.meshgrid(tmesh_w, tmesh_h)

    def _cal_sample_meshgrid(self):
        tresized_h, tresized_w = self._template_resized_bgr.shape[:2]
        scaled_tmesh_w = np.linspace(0, tresized_w - 1, self._mesh_csize)
        scaled_tmesh_h = np.linspace(0, tresized_h - 1, self._mesh_rsize)
        scaled_tmeshgrid_xy = np.meshgrid(scaled_tmesh_w, scaled_tmesh_h)
        scaled_smeshgrid_xy = RTPSTransformer.transform_template_meshgrid(
            scaled_tmeshgrid_xy, self._rtps_trans._global_H, self._rtps_trans._tps_weights,
            self._rtps_trans._template_mkpts_tps_, self._rtps_trans._parallex_,
            self._rtps_trans._overlap
        )
        scaled_smeshgrid_x, scaled_smeshgrid_y = scaled_smeshgrid_xy
        smesh_rsize, smesh_csize = scaled_smeshgrid_x.shape
        sample_meshgrid_x = np.zeros_like(scaled_smeshgrid_x)
        sample_meshgrid_y = np.zeros_like(scaled_smeshgrid_y)

        for smridx in range(smesh_rsize):
            for smcidx in range(smesh_csize):
                scaled_sx = scaled_smeshgrid_x[smridx, smcidx]
                scaled_sy = scaled_smeshgrid_y[smridx, smcidx]
                sx = scaled_sx / self._sample_x_scale
                sy = scaled_sy / self._sample_y_scale
                sample_meshgrid_x[smridx, smcidx] = sx
                sample_meshgrid_y[smridx, smcidx] = sy

        self._sample_meshgrid_xy =  (sample_meshgrid_x, sample_meshgrid_y)

    @staticmethod    
    def transform_template_pt(template_pt, template_xy_scales, sample_xy_scales,
                              H, tps_weights, template_mkpts_tps_,
                              parallex, overlap, K_smooth = 5):
        ## scale the template pt
        template_pt_x, template_pt_y = template_pt
        template_x_scale, template_y_scale = template_xy_scales
        scaled_template_pt_x = template_pt_x * template_x_scale
        scaled_template_pt_y = template_pt_y * template_y_scale

        ## transform the scaled template pt
        transformed_scaled_pt = RTPSTransformer.transform_template_pt(
            np.array([scaled_template_pt_x, scaled_template_pt_y]),
            H, tps_weights, template_mkpts_tps_, parallex, overlap, K_smooth
        )

        ## scale the transformed pt
        transformed_scaled_pt_x, transformed_scaled_pt_y = transformed_scaled_pt
        sample_x_scale, sample_y_scale = sample_xy_scales
        transformed_pt_x = transformed_scaled_pt_x / sample_x_scale
        transformed_pt_y = transformed_scaled_pt_y / sample_y_scale

        return transformed_pt_x, transformed_pt_y

    @staticmethod    
    def transform_template_pts(template_pts, template_xy_scales, sample_xy_scales,
                               H, tps_weights, template_mkpts_tps_,
                               parallex, overlap, K_smooth = 5):
        target_pts = []
        for template_pt in template_pts:
            target_pt = SuperRTPS.transform_template_pt(template_pt, template_xy_scales, 
                sample_xy_scales, H, tps_weights, template_mkpts_tps_, parallex, overlap, K_smooth)

            target_pts.append(target_pt)

        return target_pts

    @staticmethod
    def transform_sample_pt(sample_pt, sample_meshgrid_xy, template_meshgrid_xy):
        target_pt = transform_pt(sample_pt, sample_meshgrid_xy, template_meshgrid_xy)
        return target_pt
        
    @staticmethod
    def transform_sample_pts(sample_pts, sample_meshgrid_xy, template_meshgrid_xy):
        target_pts = []
        for sample_pt in sample_pts:
            target_pt = SuperRTPS.transform_sample_pt(sample_pt, sample_meshgrid_xy, template_meshgrid_xy)
            target_pts.append(target_pt)

        return target_pts

def remove_isopt(A, B, mconf,threshold):
    # 计算点集A和点集B之间的距离矩阵
    distances_A = cdist(A, A)
    distances_B = cdist(B, B)
    np.fill_diagonal(distances_A, threshold+10)
    np.fill_diagonal(distances_B, threshold+10)
    # 对点集A进行检查
    indices_to_remove_A = np.where(np.all(distances_A > threshold, axis=1))[0]

    # 对点集B进行检查
    indices_to_remove_B = np.where(np.all(distances_B > threshold, axis=1))[0]
    indices_to_remove = np.union1d(indices_to_remove_A,indices_to_remove_B)
    # 置信度低于0.5
    indices_to_remove = np.array([int(x) for x in indices_to_remove if mconf[x] < 0.5])
    if len(indices_to_remove) == 0:
        return A, B,mconf
    # 从点集A和点集B中删除对应的点
    A = np.delete(A, indices_to_remove, axis=0)
    B = np.delete(B, indices_to_remove, axis=0)
    mconf = np.delete(mconf, indices_to_remove, axis=0)

    return A, B, mconf

# 去掉异点，与notebook的区别在于: _B = B + expand_w
def remove_difpt(A,B, radius=100, threshold_degrees=5, expand_w = 960):
    # 计算点对的角度
    _B = B + expand_w
    diff = _B - A
    angles = np.degrees(np.arctan2(diff[:, 1], diff[:, 0]))
    angles = (angles+360)%180

    # 计算每个点的平均角度值
    average_angle = np.zeros(len(A))
    for i, a in enumerate(A):
        # 找到半径内的所有匹配点
        distances = np.linalg.norm(A - a, axis=1)
        indices_within_radius = np.where(distances <= radius)[0]
        # 半径内点少于5个：不做处理
        if len(indices_within_radius) <= 5:
            average_angle[i] = angles[i]
        # 半径内点数多于5个：取均值
        else:
            average_angle[i] = np.mean(angles[indices_within_radius])
    
    # 找到需要删除的索引
    indices_to_remove = np.where(angles > average_angle + threshold_degrees)[0]
    if len(indices_to_remove) > 0:
        print('='*50)
        for i in indices_to_remove:
            print(f"this angle to be deleted: {angles[i]}; Aver angle: {average_angle[i]}")
        print('='*50)
    # 从A和B中删除对应的点
    A_filtered = np.delete(A, indices_to_remove, axis=0)
    B_filtered = np.delete(B, indices_to_remove, axis=0)
    
    return A_filtered, B_filtered

# 利用直线检测器检测模版中的多横线区域
class multiline_region_detector(object):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(0)

    def remove_mlr(self,template_mkpts,sample_mkpts,mconf,conf_thres=0.5):
        # 检查若template_mkpts中某个点落在ringRegion里，且mconf小于conf_thres,则删除该点以及对应的sample_mkpts
        # template_mkpts：维度为（n,2）的numpy数组; self.ringRegion:矩形，[pt1,pt2]
        del_idx=[]
        for i in range(len(template_mkpts)):
            if mconf[i]>conf_thres:
                continue
            pt=template_mkpts[i]
            if pt[0]>self.ringRegion[0][0] and pt[0]<self.ringRegion[1][0] and pt[1]>self.ringRegion[0][1] and pt[1]<self.ringRegion[1][1]:
                if pt[0]>self.ringRegion[2][0] and pt[0]<self.ringRegion[3][0] and pt[1]>self.ringRegion[2][1] and pt[1]<self.ringRegion[3][1]:
                    continue
                else:
                    del_idx.append(i)
            else:
                continue
        template_mkpts_new=np.delete(template_mkpts,del_idx,axis=0)
        sample_mkpts_new=np.delete(sample_mkpts,del_idx,axis=0)
        mconf_new=np.delete(mconf,del_idx,axis=0)
        return template_mkpts_new,sample_mkpts_new,mconf_new

    def detect(self,img,expand_pixel=20): #return False or (x0,y0,x1,y1) of the region
        # transform to gray image
        image=img.copy() if len(img.shape) == 2 else cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(image)[0]
        line_count = 0
        start_pts = []
        for dline in lines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            angle = math.degrees(math.atan2((y1 - y0), (x1 - x0)))
            width = math.sqrt(((x1-x0) ** 2) + ((y1-y0)**2))
            if width >= image.shape[1]*2//5 and not 20 < abs(angle) < 160 and x0 < x1:
                # cv2.line(image, (x0, y0), (x1,y1), (0,0,255), 4, cv2.LINE_AA)
                # cv2.putText(image, str(line_count), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                line_count += 1
                start_pts.append([x0,y0,x1,y1])
        # 根据直线起点是否在一条竖线上检测多横线区域
        if len(start_pts) > 6 and self.are_points_on_same_vertical_line(start_pts, 10):
            self.ringRegion = self.expand_region(img,self.region_pts, expand_pixel)
            self.ringRegion += self.narrow_region(img,self.region_pts, expand_pixel)
            return self.ringRegion
        else:
            return False 

    def are_points_on_same_vertical_line(self,points, threshold):
        # 按x坐标排序
        sorted_points = sorted(points, key=lambda point: point[0])
        # 计算相邻点的x坐标差值
        x_diffs = [sorted_points[i + 1][0] - sorted_points[i][0] for i in range(len(sorted_points) - 1)]
        # 统计x坐标差值小于阈值的个数
        is_less_threshold = [1 if diff <= threshold else 0 for diff in x_diffs]
        max_count = 0
        count = 0
        start_index = -1
        for i in range(len(is_less_threshold)):
            if is_less_threshold[i] == 1:
                if count == 0:
                    start_index = i
                count += 1
                if count > max_count:
                    max_count = count
                    end_index = i
            else:
                count = 0        
        # 根据纵坐标排序,注意sorted_points长度比is_less_threshold长1
        sorted_points = sorted_points[end_index-max_count+2:end_index+2]
        sorted_points = sorted(sorted_points, key=lambda point: point[1])
        if max_count >= 6:
            self.region_pts = ((sorted_points[0][0],sorted_points[0][1]),(sorted_points[-1][2],sorted_points[-1][3])) # ((x0,y0),(x1,y1))
            return True
        else:
            return False

    def expand_region(self,im,pts,expand_pixel):
        # expand rectangular region (pt1,pt2) to include more pixels
        pt1,pt2 = pts
        h,w = im.shape[:2]
        expand_pixel_x = expand_pixel; expand_pixel_y = expand_pixel # for one side; 1/4 for both sides
        pt1 = (max(0,pt1[0]-expand_pixel_x),max(0,pt1[1]-expand_pixel_y))
        pt2 = (min(w,pt2[0]+expand_pixel_x),min(h,pt2[1]+expand_pixel_y))
        return [pt1,pt2]

    def narrow_region(self,im,pts,narraw_pixel):
        # narrow rectangular region (pt1,pt2) to include less pixels
        pt1,pt2 = pts
        h,w = im.shape[:2]
        narrow_pixel_x = narraw_pixel; narrow_pixel_y = narraw_pixel
        pt1 = (min(w,pt1[0]+narrow_pixel_x),min(h,pt1[1]+narrow_pixel_y))
        pt2 = (max(0,pt2[0]-narrow_pixel_x),max(0,pt2[1]-narrow_pixel_y))
        return [pt1,pt2]