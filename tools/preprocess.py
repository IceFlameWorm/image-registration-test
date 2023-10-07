import glob
import json
from tqdm import tqdm
from scipy.spatial import distance
import matplotlib.pyplot as plt

json_list = glob.glob("/home/alien_yhl/downloads/results_json/english_23952/"+"*_result.json")
print(len(json_list))

#calculate euclidean distance
dists = []
failures = []
rdi = []#ratio, distance, isGT
for js in tqdm(json_list):
    with open(js, 'r') as f:
        data = json.load(f)
    alignment_results = data["alignment_results"]
    try:
        for i in range(len(alignment_results.keys())):
            sample_mkpts = alignment_results[str(i)]["matched_features"]["sample_mkpts"]
            template_mkpts = alignment_results[str(i)]["matched_features"]["template_mkpts"]
            ratio = alignment_results[str(i)]["matched_features"]["mkpts_ratio2template"]
            isGT = alignment_results[str(i)]["isGroundTruthTemplate"]
            dist = 0
            if len(sample_mkpts) != len(template_mkpts):
                print("length not same".format(data["id"]))
                
            for j in range(len(sample_mkpts)):
                dist += distance.euclidean(sample_mkpts[j], template_mkpts[j])
                dist = dist / (j+1)
            dists.append(dist)
            rdi.append((ratio, dist, isGT))
    except:
        failures.append(data["id"])
print(len(dists))
print(len(failures))

r = [x[0] for x in rdi]; real_r = [x[0] for x in rdi if x[2] == 1]
d = [x[1] for x in rdi]; real_d = [x[1] for x in rdi if x[2] == 1]
figure, ax = plt.subplots(1,2)
ax[0,0].scatter(r,d,s = 0.2)
ax[0,1].scatter(real_r, real_d, s = 0.2)
figure.savefig("/home/alien_yhl/workspace/img_align/logs/1.jpg")
figure.show()
