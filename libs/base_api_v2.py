import copy
import os
from abc import ABC, abstractmethod
from pipeline_config_v2 import *
import boto3
import botocore

class BaseAPI(ABC):
    def __init__(self):
        """
        init function for loading models, must not have parameters,
        if you want to pass configuration, create a class and import it
        """

        self.pipeline_config = PipelineConfig
        self.debug_level = 0
        self.output_dir = PipelineConfig.TMP_DEBUG_RESULT_DIR
        # copy following part in init part fo your API
        self.name = 'base'      # give your API a simple but unique name, such as cn_essay_detection,
        self.flags = []         # all possible flags for this api, empty list if None
        self.input_format = ''  # input format for this API, choose from []
        self.output_format = '' # output format for this API, choose from []
        self.model = None



    @abstractmethod
    def load(self, flags):
        """
        load model
        """
        pass
    
    @abstractmethod
    def download_ckpts(self, flags):
        # download used checkpoints give flags
        """
        example:
        if "eng2" in flags:
            self.download_ckpt_from_s3('question_detection_eng2')
        if "math2" in flags:
            self.download_ckpt_from_s3('question_detection_math2')
        """
        pass

    def download_ckpt_from_s3(self,ckpt_name):
        s3 = boto3.resource('s3')
        aws_bucket = 'learnable-ocr-models'
        if ckpt_name not in PipelineConfig.CHECKPOINTS:
            raise ValueError("checkpoint name: {} not in config".format(ckpt_name))
        ckpt = PipelineConfig.CHECKPOINTS[ckpt_name]
        if os.path.exists(ckpt['local_pth']):
            print(f"[download ckpt]: {ckpt['local_pth']} exist in local")
        else:
            if not os.path.exists(os.path.dirname(ckpt['local_pth'])):
                os.makedirs(os.path.dirname(ckpt['local_pth']))
            try:
                print(f"[download ckpt]: downloading s3://{aws_bucket}/{ckpt['aws_pth']}")
                s3.Bucket(aws_bucket).download_file(ckpt['aws_pth'], ckpt['local_pth'])
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("[download ckpt]: checkpoint not found in s3: {}, {}".format(aws_bucket, ckpt['aws_pth']))
                else:
                    raise

    @abstractmethod
    def run(self, data, flag):
        pass

    def set_debug_info(self, debug_level, output_dir):
        self.output_dir = output_dir
        self.debug_level = debug_level






class LoadModelFailedException(Exception):
    pass


class RunModelFailedException(Exception):
    pass
