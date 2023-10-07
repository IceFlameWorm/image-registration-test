cd /data/home/yanghanlong/workspace/subspace/image-alignment-rejection/image-alignment-rejection
export OCR_HARDWARE_ENV=cpu
source activate ocr

#remember to check vis_img in run_mp.py
#remember to change core_num in run_mp.py
# python run_mp.py -v True -o /data/home/yanghanlong/results/image_align/my_model/epoch19_0511 -i /data/home/yanghanlong/data/image_retrieval_for_eval_ia/all_image_id.txt -s /data/home/yanghanlong/data/image_retrieval_for_eval_ia/image -j /data/home/yanghanlong/data/image_retrieval_for_eval_ia/anno -t /data/home/yanghanlong/data/image_retrieval_for_eval_ia/template_image

# run random_rotate sample image to check rotate match capability of mymodel v1.2
# python run_mp.py -v True -o /data/home/yanghanlong/results/image_align/my_model/v1_4_epoch60 -i /data/home/yanghanlong/data/image_retrieval_for_eval_ia/all_image_id.txt -s /data/home/yanghanlong/data/image_retrieval_for_eval_ia/image -j /data/home/yanghanlong/data/image_retrieval_for_eval_ia/anno -t /data/home/yanghanlong/data/image_retrieval_for_eval_ia/template_image

# python run.py -v True -o /data/home/yanghanlong/workspace/image-alignment-rejection/logs/test0417 -i ./test/test.txt -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## 0515, evaluate v1.4 epoch 60 on ptmap5k dataset
# python run_mp.py -v True -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_4_epoch60 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0517] evaluate v1.5 epoch 3 on ptmap5k dataset 
## superglue match_threshold from 0.5 to 0.7
# python run_mp.py -v True -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_5_epoch3 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0517] evaluate v1.5 epoch 3 on ptmap5k dataset
## superglue match_threshold from 0.5 to 0.6
# python run_mp.py -v True -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_5_epoch3_match_threshold_6 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0518] evaluate v1.5-resume epoch 1 on ptmap5k dataset
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_5_resume_epoch1 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0528] evaluate v1.5-resume epoch 15 on ptmap5k dataset
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_5_resume_epoch15 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0531] evaluate original weights by modify hyperparameters
## image size -> 480,640; config_v2,change model path;
# python run_mp.py -o /data/home/yanghanlong/results/image_align/original_model/ptmap_5k_mthres_6 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0602] evaluate v1_6 epoch 2(no distorsion)
## image size -> 960,1280; config_v2,change model path; core_num: 12
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v1_6_epoch2_no_distorsion -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0605] generate RPTS parameters to json result 5k
## image size -> 480,640; config_v2,change model path; core_num: 18; match_threshole -> 0.5;
# python run_mp.py -o /data/home/yanghanlong/results/image_align/original_model/ptmap_5k_with_rtps_params -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0609] 6.8W generate RPTS parameters to json result：
## image size -> 480,640; config_v2,change model path; core_num: 18; match_threshole -> 0.5;
# python run_mp.py -o /data/home/yanghanlong/results/image_align/original_model/all_6w_with_rtps_params -i "test/all_6w_cleaned.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0625] evaluate v2.1 epoch 17 on ptmap5k dataset
## image size -> 960,1280
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_1_epoch17 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0626] evaluate v2.1 epoch 21 on ptmap5k dataset
## image size -> 960,1280
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_1_epoch21 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0627] evaluate v2.1 epoch 32 on ptmap5k dataset
## image size -> 960,1280
## gpu cores -> 12
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_2_epoch3 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0701] evaluate v2.2 epoch 3 on all_6w
## image size -> 960,1280
## gpu cores -> 32
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/image_retrieval_for_eval_ia/v2_2_epoch3_all_6w -i ./test/all_6w_cleaned.txt -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0707] evaluate mymodel v2.2 epoch 3 on random_5k_0707
## image size -> 960,1280
## gpu cores -> 32
# python run_mp.py -o /data/home/yanghanlong/results/image_align/random_5k_0707/v2_2_epoch3 -i /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/all_image_id.txt -s /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/image -j /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/anno -t /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/template_image

# [0713] run image id from url
# python run.py -o ./logs/test/test0713 -i ./test/test.txt -v 1 -u 1

## [0707] evaluate orginial on random_5k_0707
## image size -> 480,640; config model path
# python run_mp.py -o /data/home/yanghanlong/results/image_align/random_5k_0707/original_model -i /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/all_image_id.txt -s /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/image -j /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/anno -t /data/home/yanghanlong/data/image_retrieval_for_random_5k_0707/template_image

# [0714] evaluate mymodel v2.2 epoch 3 on ptmap5k dataset
# image size -> 960,1280; config model path
# 去除孤点：1/5
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_2_epoch3_remove_isopt -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0714] evaluate mymodel v2.2 epoch 3 on ptmap5k dataset
# image size -> 960,1280; config model path
# 去除孤点：1/4
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_2_epoch3_remove_isopt_4 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0717] evaluate mymodel v2.3 epoch 5 on ptmap5k dataset
# image size -> 960,1280; config model path
# 不去除孤点:super_rtps.py
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch5 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# max kpts -> 2048: img_align.py load()
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch5_2048_kpts -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# max kpts -> 1024; expand template:left&right:0.15;bottom:0.1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch5_expanded -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0718] 去除异点 V2.3 epoch 5
# no template expanded:img_align,line 209; 
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch5_remove_difpt5 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# test 01bc6075-6219-4680-810d-86961228787f
# max_kpts -> 4096
# python run_mp.py -o logs/test/test0719 -i "test/test.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0719] evaluate mymodel v2.3 epoch 11 on ptmap5k dataset
# config model path
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch11 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0719]!!！！！super_dm.py line 27
# test ratio < 0.1 case in v2.3 epoch11
# python run_mp.py -o logs/test/test0721 -i "test/test.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# re-evaluate  v2.3 epoch 11 on ptmap5k 
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch11_sample_4096 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0721] evaluate v2.3 epoch 11 on ptmap 5k datset
# 增加自配准
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch11_self_match -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0724] evaluate v2.3 epoch 24 on ptmap 5k datset
# 增加自配准;config model path
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/ptmap_5k/my_model/v2_3_epoch24 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

# [0726] evaluate v2.3 epoch 24 on multiline badcase 3782
# config model path
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/multiline_badcase_3782/v2_3_epoch24 -i "test/multiline_badcase_3782.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# evaluate v2.3 epoch 28 on multiline badcase 3782
# python run_mp.py -o /data/home/yanghanlong/results/image_align/my_model/multiline_badcase_3782/v2_3_epoch28 -i "test/multiline_badcase_3782.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0810] evaluate v2.5.4 epoch 14 on ptmap 5k datset
# config model path
# 开启孤点且置信度低策略
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v2_5_4_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0810] evaluate v2.6 epoch 109 on ptmap 5k datset
# config model path
# 开启孤点且置信度低策略
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v2_6_epoch109 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0810] evaluate v2.3 epoch 35 on ptmap 5k datset
# config model path
# 开启孤点且置信度低策略；开启多横线区域过滤策略
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v2_3_epoch35 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/multiline_badcse_3782/v2_3_epoch35 -i "test/multiline_badcase_3782.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0816] evaluate v2.3 epoch 28 on 答题卡
# config model path
# 开启孤点且置信度低策略；开启多横线区域过滤策略
# export CUDA_VISIBLE_DEVICES=-1
# # #english
# allpaths=("2020zk-wz-yy") # "2020zk-hz-yy" "2020zk-jh-yy" "2020zk-jx-yy")
# for p in "${allpaths[@]}"
# do 
#     python run_mp.py -o /data/home/yanghanlong/results/image_align/answer_sheet/v2_3_epoch28/0816/english/$p -i /data/home/yanghanlong/data/答题卡/samples/英语120份/$p/image_id.txt -s /data/home/yanghanlong/data/答题卡/samples/英语120份/$p -j /data/home/yanghanlong/data/答题卡/samples/英语120份/$p/anno -t /data/home/yanghanlong/data/答题卡/templates
# done
# math
# allpaths=("2020zk-wz-math" "2020zk-hz-math" "2020zk-jh-math" "2020zk-jx-math")
# for p in "${allpaths[@]}"
# do 
#     python run_mp.py -o /data/home/yanghanlong/results/image_align/answer_sheet/v2_3_epoch28/0816/math/$p -i /data/home/yanghanlong/data/答题卡/samples/数学120份/$p/image_id.txt -s /data/home/yanghanlong/data/答题卡/samples/数学120份/$p -j /data/home/yanghanlong/data/答题卡/samples/数学120份/$p/anno -t /data/home/yanghanlong/data/答题卡/templates
# done

## [0822] evaluate v_0811_2048 on ptmap 5k
# config model path; superpoint_2 -> 2048(img_align load())
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v0811_2048 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0824] evaluate v_0811_delete_selfmatchpt on ptmap 5k
# config model path; superpoint_2 -> 2048(img_align load())
# 自配准失败的点不参与匹配
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v0811_delete_selfmatchpt -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0824] evaluate v3.5.5 on ptmap 5k
# config model path; superpoint_2 -> 2048(img_align load())
# 自配准失败的点不参与匹配
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v3_5_5_epoch10 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# epoch 18
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v3_5_5_epoch18 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0912] v3.5.1 epoch 31
# config model path; 
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v3_5_1_epoch31 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0919] evaluate v3.5.6 epoch 14
# config model path; 
# export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/v3_5_6_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# image retrieval_ia rotate
# python run_mp.py -o /data/home/yanghanlong/results/image_align/image_rotate_5000/v3_5_6_epoch14 -i /data/home/yanghanlong/data/image_retrieval_for_eval_ia/all_image_id.txt -s /data/home/yanghanlong/data/image_retrieval_for_eval_ia/image_random_rotate2 -j /data/home/yanghanlong/data/image_retrieval_for_eval_ia/anno -t /data/home/yanghanlong/data/image_retrieval_for_eval_ia/template_image
# ptmap 5k rotate 90
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate_90/v3_5_6_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/90 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate_180/v3_5_6_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/180 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate_270/v3_5_6_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/270 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image

## [0925] evaluate v3.5.7 epoch 14
# config model path; 
export CUDA_VISIBLE_DEVICES=-1
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_270/v3_5_7_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/270 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_90/v3_5_7_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/90 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_180/v3_5_7_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/180 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_0/v3_5_7_epoch14 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
# evaluate v3.5.8 epoch 9
python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_270/v3_5_8_epoch9 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/270 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_90/v3_5_8_epoch9 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/90 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_180/v3_5_8_epoch9 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/ptmap_5k_rotate/180 -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
python run_mp.py -o /data/home/yanghanlong/results/image_align/ptmap_5k/my_model/rotate/rotate_0/v3_5_8_epoch9 -i "test/5k_existed_ptmap_id.txt" -s /data/home/yanghanlong/data/image_retrieval/image -j /data/home/yanghanlong/data/image_retrieval/anno_cleaned -t /data/home/yanghanlong/data/image_retrieval/template_image
