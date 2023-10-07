import glob
import shutil
import os
from os import path as osp


import json
from tqdm import tqdm
from scipy.spatial import distance


#extract all matches img and name them by the img id
def extract_matches_rename(i_path = "test_math_vis",
                            o_path = None):
    if o_path == None:
        o_path = i_path+"_matches"
    input_path = "/home/alien_yhl/workspace/img_align/logs/"+i_path+"/**/matches.*"
    match_list = glob.glob(input_path, recursive=True)
    print("Totally {} images.".format(len(match_list)))
    if len(match_list) == 0:
        print("no images to handle!")
        return
    for ma in tqdm(match_list):
        page_no = osp.basename(osp.dirname(ma))
        x = ma.split('/')
        id = x[-3]
        name = id+"_"+page_no+'.jpg'
        output_dir = os.path.join("/home/alien_yhl/workspace/img_align/logs", o_path)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(ma, os.path.join(output_dir, name))
    print("Done!")

#save data
def save_data(rdi, path):
    rdi = ['\t'.join(map(str, x)) for x in rdi]
    with open(path, 'w') as f:
        f.writelines('\n'.join(rdi))
        
#preprocess results json file, and output rdi --> list
def preprocess_results():

    # json_list = glob.glob("/home/alien_yhl/data/results_6w/"+"*_result.json")
    json_list = glob.glob("/home/alien_yhl/downloads/results_json/math_15460/"+"*_result.json")
    # json_list = glob.glob("/home/alien_yhl/downloads/results_json/english_23952/"+"*_result.json")
    print("Totally {} json files.".format(len(json_list)))

    #calculate euclidean distance
    dists = []
    failures = []
    special = []
    rdi = []
    for js in tqdm(json_list):
        with open(js, 'r') as f:
            data = json.load(f)
        alignment_results = data["alignment_results"]
        try:
            mark = 0
            for i in alignment_results.keys():
                sample_mkpts = alignment_results[i]["matched_features"]["sample_mkpts"]
                template_mkpts = alignment_results[i]["matched_features"]["template_mkpts"]
                ratio = alignment_results[i]["matched_features"]["mkpts_ratio2template"]
                isGT = alignment_results[i]["isGroundTruthTemplate"]
                mkpts_conf = alignment_results[i]["matched_features"]["mkpts_conf"]
                aver_conf = sum(mkpts_conf) / len(mkpts_conf)
                
                if isGT == 1:
                    mark += 1
                conf = alignment_results[i]["matched_features"]["mkpts_conf"]
                dist = 0; weighted_dist = 0;
                if len(sample_mkpts) != len(template_mkpts):
                    print("length not same".format(data["id"]))
                    
                for j in range(len(sample_mkpts)):
                    
                    dist += distance.euclidean(sample_mkpts[j], template_mkpts[j])
                    weighted_dist += distance.euclidean(sample_mkpts[j], template_mkpts[j])*conf[j]
                    dist = dist / (j+1); weighted_dist = weighted_dist / (j+1);

                dists.append(dist)
                rdi.append((ratio, dist, isGT, data["id"], weighted_dist, aver_conf, i)) ####MOST IMPORTANT!
                return rdi
            
            if mark > 1:
                special.append(data["id"])
        except:
            failures.append(data["id"])
    print(len(rdi))
    print(len(failures))


    real_rdi = [x for x in rdi if x[2] == 1]
    fake_rdi = [x for x in rdi if x[2] == 0]

if __name__ == "__main__":
    extract_matches_rename("math_fn_ratio0.1")
    