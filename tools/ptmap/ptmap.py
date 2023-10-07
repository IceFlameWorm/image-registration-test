import cv2
import glob
import json
from matplotlib.patches import ConnectionPatch
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy.spatial import distance
from multiprocessing import Pool

def find_pts_with_threshold(img):
    pixel_list = []
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    for y in range(thresh.shape[0]):
        for x in range(thresh.shape[1]):
            if thresh[y, x] == 0:
                pixel_list.append([x, y])
    return pixel_list

def draw_pts(pts = [([300,300], [253,313])]):  #(tem_pt , sam_pt)
    for i, pt in enumerate(pts):
        #change np.image coords to plt.plot coords
        # pt[0][1], pt[0][0] = pt[0][0], pt[0][1]
        # pt[1][1], pt[1][0] = pt[1][0], pt[1][1]

        ax[0].plot(pt[0][0], pt[0][1], 'r.')
        ax[0].text(pt[0][0], pt[0][1]*0.97, str(i+1) )
        ax[1].plot(pt[1][0], pt[1][1], 'r.')
        ax[1].text(pt[1][0], pt[1][1]*0.97, str(i+1) )
        
        con = ConnectionPatch(pt[0], pt[1], coordsA = "data", coordsB = "data",axesA= ax[0], axesB = ax[1])
        fig.add_artist(con)

def find_mpt(tem_pt, H):
    #change np.image coords to plt.plot coords
    # tem_pt[0], tem_pt[1] = tem_pt[1], tem_pt[0]
    
    _tem_pt = list(tem_pt)
    _tem_pt.append(1)
    _tem_pt = np.array(_tem_pt)
    H = np.array(H)
    # tem_pt = tem_pt.reshape(-1,1)
    sam_pt = H @ _tem_pt
    sam_pt = list(sam_pt[0:2]/sam_pt[2])
    return (tem_pt, sam_pt)

def find_gt_template(id):
    jpath = glob.glob(f"/data/home/yanghanlong/data/image_retrieval/anno_cleaned/{id}.json")
    with open(jpath[0], 'r') as f:
        dt = json.load(f)
    tem_path = dt["gt_template_image"]
    if tem_path == "":
        return (-1, tem_path)
    else:
        return (cv2.imread(osp.join("/data/home/yanghanlong/data/image_retrieval/template_image", tem_path)), tem_path)
    
def find_gt_index(data):
    for k in data["alignment_results"]:
        if data["alignment_results"][k]["isGroundTruthTemplate"] == 1:
            return k
    return -1
def find_gt_H(data, k):
    return data["alignment_results"][k]["alignment_params"]["H"]

def pt_scale2ori( mpts,data, k):
    tem_scale = data["alignment_results"][k]["alignment_params"]["template_xy_scales"]
    sam_scale = data["alignment_results"][k]["alignment_params"]["sample_xy_scales"]
    _mpts = []
    for pt in mpts:
        pt[0][0] /= tem_scale[0]
        pt[0][1] /= tem_scale[1]
        pt[1][0] /= sam_scale[0]
        pt[1][1] /= sam_scale[1]
        _mpts.append(pt)
    return _mpts

def find_tem_pts(tem, num_pts, dxy = [320, 240], H = 1):
    array_H = np.array(H)
    inverse_H = np.linalg.inv(array_H)

    black_pixels = []
    black_pixels = [[x+dxy[0], y+ dxy[1]] for x, y in find_pts_with_threshold(tem)]

    # for i in range(0, tem.shape[0]-1, 1):
    #     for j in range(0, tem.shape[1]-1, 1):
    #         if tem[i][j].sum() < 100:
    #             black_pixels.append([j+dxy[0], i+dxy[1]])
    #         else:
    #             pass
                
    _black_pixels = list(black_pixels)
    for  b_pixel in _black_pixels:
        _tem_pt = find_mpt(b_pixel, H)[1]
        if (0 < _tem_pt[0] < 478) and (0 < _tem_pt[1] < 638): #delete pts out of bound
            continue
        else:
           black_pixels.remove(b_pixel)

    num_allpts = len(black_pixels)
   
    if num_allpts == 0:
        return []
    elif num_allpts < num_pts:  ##pts num is too small
        if len(black_pixels)>1 and distance.euclidean(black_pixels[0], black_pixels[-1]) < 30: #check if pts too close
            black_pixels = black_pixels[0:1]
        try: #check if pts too close
            while distance.euclidean(black_pixels[0], black_pixels[1]) < 10:
                black_pixels.pop(0)
            if distance.euclidean(new_black_pixels[-2], new_black_pixels[-3]) < 30:
                new_black_pixels.pop(-2)
        except:
            pass
        return [find_mpt(x, H) for x in black_pixels]
    #find middle points
    new_black_pixels = []
    x = np.linspace(1, num_allpts, num_pts)
    for i in x: 
        j = i 
        
        new_black_pixels.append(black_pixels[int(j-1)])
    if distance.euclidean(new_black_pixels[0], new_black_pixels[-1]) < 30: #check if pts too close
        new_black_pixels = new_black_pixels[0:1]
    elif distance.euclidean(new_black_pixels[-2], new_black_pixels[-1]) < 30: #check if pts too close
        new_black_pixels.pop()
    try: #check if pts too close
        while distance.euclidean(new_black_pixels[0], new_black_pixels[1]) < 30:
            new_black_pixels.pop(0)
    except:
        pass
    mpts = [find_mpt(x, H) for x in new_black_pixels]

    return mpts

#directory change
#run with multiprocess
jpath_list = glob.glob("/data/home/yanghanlong/results/iar_on_rule/cleaned_dataset/6w_warped_sam/*.json")
print("="*30)
print(f"Total file num: {len(jpath_list)}")                
print("="*30)

def job(js):
    with open(js, 'r') as f:
        data = json.load(f)
    im_id = data["id"]
    ##return if json file already exists
    if osp.exists("/data/home/yanghanlong/results/pt_map_v3/results/{}.json".format(im_id)):
        return
    gt_index = find_gt_index(data)
    if gt_index == -1:
        with open("/data/home/yanghanlong/results/pt_map_v3/no_gt/no_gt_id.txt", 'a') as f:
            f.write('\n')
            f.write(im_id)
        return
    else:
        H = find_gt_H(data, gt_index)
    tem_data = find_gt_template(im_id) ####
    tem = tem_data[0]
    ##tem == -1 will throw error if tem is a image, tem.any() == -1 will throw error if tem is -1; so use try, except
    try:
        if tem == -1:
            with open("/data/home/yanghanlong/results/pt_map_v3/no_gt/not_find_tem.txt", 'a') as f:
                f.write('\n')
                f.write(im_id)
            return
    except:
        pass
    o_data = {}

    tem_rgb = cv2.cvtColor(tem, cv2.COLOR_BGR2RGB)
    tem_scale = cv2.resize(tem_rgb, (480, 640))

    # find match pts and draw them
    h,w = tem_scale.shape[0], tem_scale.shape[1]

    mpts = find_tem_pts(tem_scale[0:h//2,0:w//2, :], 4, dxy = [0,0], H=H)
    mpts += find_tem_pts(tem_scale[0:h//2,w//2+20:w, :], 4, dxy = [260,0], H=H)
    mpts += find_tem_pts(tem_scale[h//2+10:h,0:w//2, :], 4, dxy = [0, 330], H=H)
    mpts += find_tem_pts(tem_scale[h//2+10:h,w//2+20:w, :], 4, dxy = [260, 330], H=H)



    mpts = pt_scale2ori(mpts, data, gt_index)
    mpts = sorted(mpts, key=lambda x: (x[0][1], x[0][0]))
    template_pts = [[x[0][0], x[0][1]]for x in mpts]
    sample_pts = [[x[1][0], x[1][1]]for x in mpts]
    o_data = {
        "image_id": im_id,
        "template_path":tem_data[1],
        "num_mpts": len(template_pts),
        "template_pts": template_pts,
        "sample_pts": sample_pts
    }
    with open("/data/home/yanghanlong/results/pt_map_v3/results/{}.json".format(im_id), 'w') as f:
        json.dump(o_data, f)
    return

with Pool(30) as p:
    p.map(job, tqdm(jpath_list))

#print results info
suc =  glob.glob("/data/home/yanghanlong/results/pt_map_v3/results/*.json")
with open("/data/home/yanghanlong/results/pt_map_v3/no_gt/no_gt_id.txt", 'r') as f:
    fai = f.readlines()
print("="*30)
print(f"Successful file num: {len(suc)}")  
print(f"Failure file num: {len(fai)-1}")  
print(f"Total: {len(fai)-1+len(suc)}")  
print("="*30)
    