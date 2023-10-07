import matplotlib.cm as cm
from .models.matching import Matching,Matching_custom_max_kpts
from .models.utils import convert_image, make_matching_plot_fast


class SuperDM(object):
    default_config = {
        'superpoint': {
            'num_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'weights_file': ''
        },
        'superglue': {
            'sinkhorn_iterations': 20,
            'match_threshold': 0.5,
            'weights_file': ''
        }
    }

    def __init__(self, superpoint_config = {}, superglue_config = {}, device = 'cpu'):
        self._config = {**self.default_config}
        self._config['superpoint'].update(superpoint_config)
        self._config['superglue'].update(superglue_config)

        self._device = device
        self._matching_model = Matching_custom_max_kpts(self._config).eval().to(device)

    def detect_match(self, img_bgr_0, img_bgr_1):
        image0, inp0 = convert_image(img_bgr_0, self._device)
        image1, inp1 = convert_image(img_bgr_1, self._device)
        # print(inp1[:,:,350,350])
        # if inp1[:,:,350,350] - 0.7451 < 0.0001:
        #     print('='*30)
        #     import cv2
        #     import numpy as np
        #     im = inp1[0,0,:,:].numpy()
        #     im = (im * 255).astype(np.uint8)
        #     cv2.imwrite("/data/home/yanghanlong/2.jpg", im)
            
        #     open("/data/home/yanghanlong/1.txt", 'w').writelines(inp0.tolist())
        pred = self._matching_model({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        image0_0_matches = pred['image0_0_matches']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        image0_0_matches_num = (image0_0_matches > -1).sum()

        out_matches = {'kpts0': kpts0, 'kpts1': kpts1,
                       'matches': matches, 'conf': conf,
                       'mkpts0': mkpts0, 'mkpts1': mkpts1, 'mconf': mconf,
                       'img_gray_0': image0, 'img_gray_1': image1,
                       'image0_0_matches_num':image0_0_matches_num
                      }

        return out_matches

    @staticmethod
    def vis_matches(out_matches, viz_path = None, show_keypoints = False):
        mconf = out_matches['mconf']
        kpts0, kpts1 = out_matches['kpts0'], out_matches['kpts1']
        mkpts0, mkpts1 = out_matches['mkpts0'], out_matches['mkpts1']
        img_gray_0, img_gray_1 = out_matches['img_gray_0'], out_matches['img_gray_1']
        image0_0_matches_num = out_matches["image0_0_matches_num"]
        color = cm.jet(mconf)
        text = [
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
            f'Ratio: {len(mkpts0)/image0_0_matches_num:.3f} = {len(mkpts0)}/{image0_0_matches_num}'
        ]
        matches_img_bgr = make_matching_plot_fast(img_gray_0, img_gray_1, kpts0, kpts1,
            mkpts0, mkpts1, color, text, viz_path, show_keypoints
        )

        return matches_img_bgr