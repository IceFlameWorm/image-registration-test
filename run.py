from libs.img_align import ImgAlignAPI

from glob import glob
from os.path import join, basename, dirname, exists, splitext, abspath
import argparse
import sys
from libs.ops.simple_pipeline import SimplePipeline
import asyncio
import json
import requests
from p_tqdm import p_umap
from pprint import pprint
from pipeline_config_v2 import PipelineConfig
import pandas
from tqdm import tqdm
import os
from time import time
import traceback
import random





if __name__ == '__main__':
    # #run with CPU
    # os.environ["OCR_HARDWARE_ENV"] = "cpu"

    parser = argparse.ArgumentParser(description='Please set input folder and flags for your test run')
    parser.add_argument('-i','--input_txt_path',default="./test/all_6w_id.txt",help='img to run a folder of images, [raw] to run a of raw request jsons')
    parser.add_argument('-s','--sample_img_path',default="/data/home/yanghanlong/data/image_retrieval/image",help='img to run a folder of images, [raw] to run a of raw request jsons')
    parser.add_argument('-j','--anno_json_path',default="/data/home/yanghanlong/data/image_retrieval/anno",help='img to run a folder of images, [raw] to run a of raw request jsons')
    parser.add_argument('-t','--template_img_path',default="/data/home/yanghanlong/data/image_retrieval/template_image",help='img to run a folder of images, [raw] to run a of raw request jsons')
    parser.add_argument('-o', '--output_results_folder', default="/data/home/yanghanlong/results/iar_on_rule/all_6w_get_wraped_sam")
    parser.add_argument('-v', '--img_visual', type=int,default=0)
    parser.add_argument('-u', '--run_single_url', type=int,default=0)
    args = parser.parse_args()

    img_align_api = ImgAlignAPI(args)
    img_align_api.load(['multi', 'page_select'])
    print(args)

    if args.run_single_url:
        img_align_api.run_url(args.input_txt_path, vis_img=args.img_visual)
    else:
        img_align_api.run(args.input_txt_path, vis_img=args.img_visual)
        
    

