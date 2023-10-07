from libs.base_api_v2 import *
from libs.img_align_src.feature_dm.super.super_dm import SuperDM
from libs.img_align_src.pipeline.super_rtps import SuperRTPS
from libs.img_align_src.transform.rtps.rtps_transform import RTPSTransformer
from libs.img_align_src.utils.visualize import draw_meshgrid
from libs.ops.logging_ops import *

import numpy as np
import cv2
import requests
import time
import os.path as osp
import json
from tqdm import tqdm
import shutil
from multiprocessing import Pool, Process
#multiprocess
from p_tqdm import p_umap
from functools import partial


def get_raw_request(url):
            data = None
            # retry
            fetch_succeed = False
            count = 3
            while count > 0:
                try:
                    raw_response = requests.get(url)
                    if raw_response.status_code == 200:
                        data = raw_response.json()['data']
                        fetch_succeed = True
                        break
                except:
                    traceback.print_exc()
                    pass
                count -= 1
            if not fetch_succeed:
                print(f"failed to get raw response with this url: {url}")
            return data

def download_img(url):
    max_try_num = 5
    try_num = 0
    timeout = 20
    try_interval = 0.1 # seconds

    img = None
    while try_num < max_try_num:
        try:
            image_data = requests.get(url, timeout= (timeout, timeout)).content
            
            img_np = np.fromstring(image_data, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            info_msg = gen_logging_string('img_align_api', 'download_img', 'downloading success',
                    f'{try_num + 1}th try: succeeded to download template image from: {url}')
            logging.info(info_msg)
            break
        except Exception as e:
            warning_msg = gen_logging_string('img_align_api', 'download_img', 'downloading failure',
                    f'{try_num + 1}th try: failed to download template image from: {url}. Warning msg: f{e}')
            logging.warning(warning_msg)
            time.sleep(try_interval)
            try_num += 1
    
    return img


def run_once( name_list, flags, vis_img,args):
    api = ImgAlignAPI(args)
    api.load()
    api.vis_img = vis_img
    for name in name_list:
        # try:
        data = api.run_single(name, flags)
        # except:
        #     print(f"[FAILURE] {name}")
        #     continue
        with open(osp.join(args.output_results_folder, name+"_result.json"), 'w') as f:
            json.dump(data, f)


def run_mp(input_txt_path, args, flags = ['multi', 'page_select'], vis_img = True):
    '''
    Args:
        input_txt_path: 
        flags: 
    '''
    ##read input_txt_path and extract name list
    vis_img = args.img_visual
    with open(input_txt_path, 'r') as f:
        name_list = f.readlines()
        name_list = [ x.strip() for x in name_list]
    #make output dir
    os.makedirs(args.output_results_folder, exist_ok=True)

    split_num = 55 #split to how many slices, generally = cpu cores
    batches = []
    for i in range(0, len(name_list), len(name_list)//split_num):
        batches.append(name_list[i:i+len(name_list)//split_num])
    p_umap(partial(run_once, flags = flags, vis_img = vis_img, args = args), batches)
    
    


class ImgAlignAPI(BaseAPI):
    def __init__(self, args):
        super(ImgAlignAPI, self).__init__()
        self.name = 'img_align'
        self.args = args
    def download_ckpts(self, flags):
        self.download_ckpt_from_s3('img_align_kp_det')
        self.download_ckpt_from_s3('img_align_kp_match')

    def load(self, flags= ['multi', 'page_select']):
        superpoint_config = {
            # 'weights_file': PipelineConfig.CHECKPOINTS['img_align_kp_det']['local_pth']
            'weights_file': PipelineConfig.CHECKPOINTS['img_align_kp_det_mymodel']['local_pth'],
            'image0_max_kpts':1024,
            'image1_max_kpts':2048
        }
        superglue_config = {
            # 'weights_file': PipelineConfig.CHECKPOINTS['img_align_kp_match']['local_pth']
            'weights_file': PipelineConfig.CHECKPOINTS['img_align_kp_match_mymodel']['local_pth']
            # 'match_threshold':0.6

        }

        use_cpu = os.environ.get('OCR_HARDWARE_ENV')=='cpu'
        device = 'cpu' if use_cpu else 'cuda'
        self._fdm = SuperDM(superpoint_config= superpoint_config, superglue_config= superglue_config, device=device)

    def run_url(self, input_txt_path, flags = ['multi', 'page_select'], vis_img = 0):
        #read self.input_txt_path
        with open(input_txt_path, 'r') as f:
            self.name_list = f.readlines()
            
        #make output dir
        os.makedirs(self.args.output_results_folder, exist_ok=True)
        #run_single and save results
        with tqdm(self.name_list) as bar:
            for name in bar:
                name = name.strip()
                bar.set_postfix(name = name)
                data = self.run_single_url(name, flags)
                if data == None:
                    continue
                with open(osp.join(self.args.output_results_folder, name+"_result.json"), 'w') as f:
                    json.dump(data, f)

    def run(self, input_txt_path, flags = ['multi', 'page_select'], vis_img = False):
        '''
        Args:
            input_txt_path: 
            flags: 
        '''
        if vis_img == 'True':
            self.vis_img = True
        elif vis_img == 'False':
            self.vis_img = False
        else:
            self.vis_img = vis_img
        with open(input_txt_path, 'r') as f:
            self.name_list = f.readlines()
        #make output dir
        os.makedirs(self.args.output_results_folder, exist_ok=True)
        #run_single and save results
        with tqdm(self.name_list) as bar:
            for name in bar:
                name = name.strip()
                bar.set_postfix(name = name)
                data = self.run_single(name, flags)
                with open(osp.join(self.args.output_results_folder, name+"_result.json"), 'w') as f:
                    json.dump(data, f)
                

    def run_single(self, name, flags = ['multi', 'page_select']):
        if osp.exists(osp.join(self.args.output_results_folder, name+"_result.json")):
            return json.load(open(osp.join(self.args.output_results_folder, name+"_result.json"),'r'))
        data = {}
        data['id'] = name
        data['alignment_results'] = {}
        ## get the sample and template images
        sample_bgr = self._get_sample_img(name)
        template_imgs = self._get_template_imgs_2(name, flags)

        ## align the sample and the template images
        rtps_config = {}
        mesh_rsize, mesh_csize = 31, 31
        img_width, img_height = 960, 1280

        failures = []

        # if self.debug_level >= 1:
        #     debug_path = self.output_dir
        #     template_imgs_dir = os.path.join(debug_path, 'template_imgs')
        #     os.makedirs(template_imgs_dir, exist_ok= True)
        fail_num = 0 #record fail, distinguish fail_part or fail_whole
        best_matchs = 0
        
        for page_no, template_bgr, is_gt_template, tem_path in template_imgs:
            # print(template_bgr.shape)
            _h, _w = template_bgr.shape[:2]
            y_top_border_rate = 0
            y_bottom_border_rate = 0.1
            x_border_rate = 0.15
            y_top_border = int(_h * y_top_border_rate)
            y_bottom_border = int(_h * y_bottom_border_rate)
            x_border = int(_w * x_border_rate)
            # 是否expand
            # template_bgr = cv2.copyMakeBorder(template_bgr, y_top_border, y_bottom_border, x_border, x_border, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

            ## save template img
            # if self.debug_level >= 1:
            #     template_img_file = os.path.join(template_imgs_dir, f'template_{page_no}.jpg')
            #     cv2.imwrite(template_img_file, template_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])

            ## align the sample and template images
            super_rtps = SuperRTPS(self._fdm, rtps_config, template_bgr, sample_bgr, resize= [img_width, img_height],
                mesh_rsize= mesh_rsize, mesh_csize= mesh_csize)
            # super_rtps.align()
            try:  
                super_rtps.align()
            except Exception as e:
                fail_num += 1
                fail_part_dir = osp.join(self.args.output_results_folder, "failure","fail_part", name)
                os.makedirs(fail_part_dir, exist_ok=True)
                cv2.imwrite(osp.join(fail_part_dir, f'template_{page_no}_isGT_{is_gt_template}.jpg'), template_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
                cv2.imwrite(osp.join(fail_part_dir, f'sample.jpg'), sample_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
                continue

            ## save the align results
            alignment_params = {
                'template_xy_scales': (super_rtps._template_x_scale, super_rtps._template_y_scale),
                'sample_xy_scales': (super_rtps._sample_x_scale, super_rtps._sample_y_scale),
                'H': super_rtps._rtps_trans._global_H.tolist(),
                'tps_weights': {k : v.tolist() for k,v in super_rtps._rtps_trans._tps_weights.items()},
                'template_mkpts_tps_': super_rtps._rtps_trans._template_mkpts_tps_.tolist(),
                'parallex': super_rtps._rtps_trans._parallex_.tolist(),
                'overlap': {k : v for k,v in super_rtps._rtps_trans._overlap.items()}, 
                'K_smooth': 5
            }

            #add or delete results here
            matched_features = {

                'sample_mkpts': super_rtps._feature_matches['mkpts1'].tolist(),
                'template_mkpts': super_rtps._feature_matches['mkpts0'].tolist(),
                'mkpts_conf': super_rtps._feature_matches['mconf'].tolist(),
                'mkpts_count': len(super_rtps._feature_matches['mkpts1']),
                'mkpts_ratio2template': len(super_rtps._feature_matches['mkpts1'])/len(super_rtps._feature_matches['kpts0']),
                'mkpts_ratio2sample': len(super_rtps._feature_matches['mkpts1'])/len(super_rtps._feature_matches['kpts1']),
                'ratio_modify_by_image0_0':len(super_rtps._feature_matches['mkpts1'])/super_rtps._feature_matches['image0_0_matches_num']
            }

            template_features = {
                'tem_shape': list(template_bgr.shape),
                'resized_tem_shape':list(super_rtps._template_resized_bgr.shape),
                'template_kpts_count': len(super_rtps._feature_matches['kpts0']),
                'template_kpts': super_rtps._feature_matches['kpts0'].tolist(),
                'kpts_match': super_rtps._feature_matches['matches'].tolist(),
                'kpts_conf': super_rtps._feature_matches['conf'].tolist()
            }

            sample_features = {
                'sam_shape': list(sample_bgr.shape),
                'resized_sam_shape':list(super_rtps._sample_resized_bgr.shape),
                'sample_kpts_count': len(super_rtps._feature_matches['kpts1']),
                'sample_kpts': super_rtps._feature_matches['kpts1'].tolist()
            }

            warp_params = {
                'sample_meshgrid_xy': super_rtps._sample_meshgrid_xy,
                'template_meshgrid_xy': super_rtps._template_meshgrid_xy
            }


            data['alignment_results'][page_no] = {
                'isGroundTruthTemplate': is_gt_template,
                'template_path': tem_path,
                'alignment_params': alignment_params,
                'matched_features': matched_features,
                'template_features': template_features,
                'sample_features': sample_features
                # 'warp_params': warp_params
            }

            ## save the debug results
            if self.vis_img == True:
                self._save_debug_res(name, page_no, sample_bgr, template_bgr, super_rtps, is_gt_template)

            # if 'page_select' in flags:
            #     if len(matched_features['sample_mkpts']) > best_matchs:
            #         best_matchs = len(matched_features['sample_mkpts'])
            #         data['info']['struct']['page_no'] = page_no
            #         data['info']['struct']['matched'] = True

        # if 'page_select' in flags:
        #     # update page_meta
        #     page_meta = {}
        #     for question_meta in data['info']['meta']:
        #         for subquestion_meta in question_meta['questionSubList']:
        #             question_key = str(question_meta['questionId']) + '_0'
        #             if question_key not in page_meta:
        #                 page_meta[question_key] = {}
        #                 page_meta[question_key]['questionId'] = question_meta['questionId']
        #                 page_meta[question_key]['displayQuestionNumber'] = question_meta['displayQuestionNumber']
        #                 page_meta[question_key]['questionBody'] = question_meta['questionBody']
        #                 page_meta[question_key]['isMachineGrade'] = question_meta['isMachineGrade']
        #                 page_meta[question_key]['secondLevelQuestionType'] = question_meta['secondLevelQuestionType']
        #             subquestion_key = str(question_meta['questionId']) + '_' + str(subquestion_meta['bookPageQuestionId'])
        #             if subquestion_key not in page_meta:
        #                 page_meta[subquestion_key] = copy.deepcopy(subquestion_meta)
        #     data['info']['page_meta'] = page_meta

            # update page match
            # if 'page_no' not in data['info']['struct']:
            #     data['info']['struct']['page_no'] = -1
            #     data['info']['struct']['matched'] = False

        ## logs
        # aligned_template_pages = list(data['alignment_results'].keys())
        # logging.info(
        #     gen_logging_string('img_align_api', data['id'], 'success',
        #         f'aligned template pages: {aligned_template_pages}'
        #     )
        # )

        # if failures != []:
        #     aligned_template_pages = list(data['alignment_results'].keys())
        #     logging.warning(
        #         gen_logging_string('img_align_api', data['id'], 'failues',
        #             f'alignment failues : {failures}'
        #         )
        #     )
 
        #make fail_whole_dir
        fail_whole_dir = osp.join(self.args.output_results_folder, "failure","fail_whole")
        os.makedirs(fail_whole_dir, exist_ok=True)
        if fail_num == len(template_imgs) and fail_num != 0:
            shutil.move(fail_part_dir, fail_whole_dir)

        return data

    def run_single_url(self, name, flags = ['multi', 'page_select']):
        #get json file
        dt = get_raw_request(PipelineConfig.GET_PREPROD_REQUEST_FROM_ID_URL.format(name))

        sample_url = dt["img_url"]
        template_urls = [(x["pageNo"], x["pageUrl"], 0) for x in dt["question_position_info"]]

        if osp.exists(osp.join(self.args.output_results_folder, name+"_result.json")):
            return json.load(open(osp.join(self.args.output_results_folder, name+"_result.json"),'r'))
        
        data = {}
        data['id'] = name
        data['alignment_results'] = {}
        ## get the sample and template images
        sample_bgr = download_img(sample_url)
        template_imgs = [(x[0], download_img(x[1]), x[2]) for x in template_urls]
        ## align the sample and the template images
        rtps_config = {}
        mesh_rsize, mesh_csize = 31, 31
        img_width, img_height = 960, 1280
        fail_num = 0 #record fail, distinguish fail_part or fail_whole
        
        for page_no, template_bgr, is_gt_template in template_imgs:
            # print(template_bgr.shape)
            _h, _w = template_bgr.shape[:2]
            y_top_border_rate = 0
            y_bottom_border_rate = 0
            x_border_rate = 0
            y_top_border = int(_h * y_top_border_rate)
            y_bottom_border = int(_h * y_bottom_border_rate)
            x_border = int(_w * x_border_rate)
            template_bgr = cv2.copyMakeBorder(template_bgr, y_top_border, y_bottom_border, x_border, x_border, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

            ## save template img
            # if self.debug_level >= 1:
            #     template_img_file = os.path.join(template_imgs_dir, f'template_{page_no}.jpg')
            #     cv2.imwrite(template_img_file, template_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])

            ## align the sample and template images
            super_rtps = SuperRTPS(self._fdm, rtps_config, template_bgr, sample_bgr, resize= [img_width, img_height],
                mesh_rsize= mesh_rsize, mesh_csize= mesh_csize)
            # super_rtps.align()
            try:
                super_rtps.align()
            except Exception as e:
                fail_num += 1
                fail_part_dir = osp.join(self.args.output_results_folder, "failure","fail_part", name)
                os.makedirs(fail_part_dir, exist_ok=True)
                cv2.imwrite(osp.join(fail_part_dir, f'template_{page_no}.jpg'), template_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
                cv2.imwrite(osp.join(fail_part_dir, f'sample.jpg'), sample_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
                continue

            ## save the align results
            alignment_params = {
                'template_xy_scales': (super_rtps._template_x_scale, super_rtps._template_y_scale),
                'sample_xy_scales': (super_rtps._sample_x_scale, super_rtps._sample_y_scale),
                'H': super_rtps._rtps_trans._global_H.tolist(),
                'K_smooth': 5
            }

            #add or delete results here
            matched_features = {

                'sample_mkpts': super_rtps._feature_matches['mkpts1'].tolist(),
                'template_mkpts': super_rtps._feature_matches['mkpts0'].tolist(),
                'mkpts_conf': super_rtps._feature_matches['mconf'].tolist(),
                'mkpts_count': len(super_rtps._feature_matches['mkpts1']),
                'mkpts_ratio2template': len(super_rtps._feature_matches['mkpts1'])/len(super_rtps._feature_matches['kpts0']),
                'mkpts_ratio2sample': len(super_rtps._feature_matches['mkpts1'])/len(super_rtps._feature_matches['kpts1'])
            }

            template_features = {
                'resized_tem_shape':list(super_rtps._template_resized_bgr.shape),
                'template_kpts_count': len(super_rtps._feature_matches['kpts0']),
                'template_kpts': super_rtps._feature_matches['kpts0'].tolist(),
                'kpts_match': super_rtps._feature_matches['matches'].tolist(),
                'kpts_conf': super_rtps._feature_matches['conf'].tolist()
            }

            sample_features = {
                'resized_sam_shape':list(super_rtps._sample_resized_bgr.shape),
                'sample_kpts_count': len(super_rtps._feature_matches['kpts1']),
                'sample_kpts': super_rtps._feature_matches['kpts1'].tolist()
            }

            warp_params = {
                'sample_meshgrid_xy': super_rtps._sample_meshgrid_xy,
                'template_meshgrid_xy': super_rtps._template_meshgrid_xy
            }


            data['alignment_results'][page_no] = {
                'isGroundTruthTemplate': is_gt_template,
                'alignment_params': alignment_params,
                # 'template_img': template_bgr,
                'matched_features': matched_features,
                'template_features': template_features,
                'sample_features': sample_features
                # 'warp_params': warp_params
            }

            ## save the debug results
            if self.vis_img == True:
                self._save_debug_res(name, page_no, sample_bgr, template_bgr, super_rtps)
        
        #make fail_whole_dir
        fail_whole_dir = osp.join(self.args.output_results_folder, "failure","fail_whole")
        os.makedirs(fail_whole_dir, exist_ok=True)
        if fail_num == len(template_imgs) and fail_num != 0:
            shutil.move(fail_part_dir, fail_whole_dir)

        return data

    def _get_sample_img(self, name):
        img_path = osp.join(self.args.sample_img_path, name+".jpg")
        sample_bgr = cv2.imread(img_path)
        return sample_bgr

    def _get_template_imgs(self, data, flags):
        '''
        Needs override
        '''

        ## if 'multi' in flags:
        ##     pass
        ## else:
        ##     pass

        # template_bgr = self._get_sample_img(data)
        # template_img_file = data['info']['template_img_file']
        # template_bgr = cv2.imread(template_img_file)
        # template_imgs = [(0, template_bgr)]

        # template_imgs = []
        # for page_no_str, template_img_file in data['info']['template_img_files'].items():
        #     page_no = int(page_no_str)
        #     template_bgr = cv2.imread(template_img_file)
        #     template_imgs.append((page_no, template_bgr))

        if 'multi' in flags:
            page_no_files = [
                (int(page_no_str), img_file)
                for page_no_str, img_file in data['info']['template_img_files'].items()
            ]
        else:
            # matched_page_no = data['info']['struct']['page_no']
            page_no_files = [
                (int(page_no_str), img_file)
                for page_no_str, img_file in data['info']['template_img_files'].items()
                    if int(page_no_str) == matched_page_no
            ]

        template_imgs = []
        for page_no, template_img_file in page_no_files:
            template_bgr = cv2.imread(template_img_file)
            template_imgs.append((page_no, template_bgr))

        return template_imgs

    def _get_template_imgs_2(self, name, flags):
       
        #read json file
        anno_path = osp.join(self.args.anno_json_path, name+".json")
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        template_imgs_path = anno["candidate_template_images"]
        gt_template_img_path = anno["gt_template_image"]
        #load template images
        page_no = 0
        template_imgs = []

        for template_img_path in template_imgs_path:
            if template_img_path == gt_template_img_path:
                is_gt_template = 1
            else:
                is_gt_template = 0
            template_img_abspath = osp.join(self.args.template_img_path, template_img_path)
            template_bgr = cv2.imread(template_img_abspath)
            # # expanded template:left&right:0.15;bottom:0.1
            # if expanded:
            #     y_top_exp = int(th * y_top_exp_rate)
            #     y_bottom_exp = int(th * y_bottom_exp_rate)
            #     x_left_exp = int(tw * x_left_exp_rate)
            #     x_right_exp = int(tw * x_right_exp_rate)
            #     template_bgr = cv2.copyMakeBorder(template_bgr, y_top_exp, y_bottom_exp, x_left_exp, x_right_exp,
            #             borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
            template_imgs.append((page_no, template_bgr, is_gt_template, template_img_path))
            page_no += 1


        return template_imgs


    def _save_debug_res(self, name, page_no, sample_bgr, template_bgr, super_rtps, is_gt_template):
        matches_img_bgr = SuperDM.vis_matches(super_rtps._feature_matches)

        warped_sample_bgr = RTPSTransformer.transform_img(
            sample_bgr, super_rtps._sample_meshgrid_xy, super_rtps._template_meshgrid_xy)

        _, blended_bgr = RTPSTransformer.blend_imgs(warped_sample_bgr, template_bgr)

        template_mesh_bgr, template_grids = draw_meshgrid(
            template_bgr.copy(), super_rtps._template_meshgrid_xy)

        sample_mesh_bgr, sample_grids = draw_meshgrid(
            sample_bgr.copy(), super_rtps._sample_meshgrid_xy, color=[0,0,255])

        # template_sample_mesh_bgr = np.concatenate([templaplate_mesh_bgr, sample_mesh_bgr], axis = 1)
        debug_save_dir = os.path.join(self.args.output_results_folder, name, f'page_{page_no}_isGT_{is_gt_template}')
        os.makedirs(debug_save_dir, exist_ok=True)

        # cv2.imwrite(os.path.join(debug_save_dir, 'sample.jpg'), sample_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        # cv2.imwrite(os.path.join(debug_save_dir, 'template.jpg'), template_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(os.path.join(debug_save_dir, 'matches.jpg'), matches_img_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(os.path.join(debug_save_dir, 'warped_sample.jpg'), warped_sample_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(os.path.join(debug_save_dir, 'blended.jpg'), blended_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        # cv2.imwrite(os.path.join(debug_save_dir, 'sample_mesh.jpg'), sample_mesh_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
        # cv2.imwrite(os.path.join(debug_save_dir, 'template_mesh.jpg'), template_mesh_bgr, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])

