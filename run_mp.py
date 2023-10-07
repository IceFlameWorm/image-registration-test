from libs.img_align import ImgAlignAPI, run_mp

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
import torch
torch.set_num_threads(1)


parser = argparse.ArgumentParser(description='Please set input folder and flags for your test run')
parser.add_argument('-i','--input_txt_path',default='./test/math_15460.txt')
parser.add_argument('-s','--sample_img_path',default='./data/image_retrieval/image',help='img to run a folder of images, [raw] to run a of raw request jsons')
parser.add_argument('-j','--anno_json_path',default='./data/image_retrieval/anno',help='img to run a folder of images, [raw] to run a of raw request jsons')
parser.add_argument('-t','--template_img_path',default='./data/image_retrieval/template_image',help='img to run a folder of images, [raw] to run a of raw request jsons')
parser.add_argument('-o', '--output_results_folder', default='./logs/test')
parser.add_argument('-v', '--img_visual', default=1)

args = parser.parse_args()



if __name__ == '__main__':
    with torch.no_grad():
        run_mp(args.input_txt_path, args, flags = ['multi', 'page_select'], vis_img = False)
    

