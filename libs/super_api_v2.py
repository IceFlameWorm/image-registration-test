import logging
import os
from os.path import join
import datetime
import re
import torch
import sys, traceback

from libs.ops.pipeline_ops import *
from pipeline_config_v2 import PipelineConfig

from libs.ops.metrics.metrics_publisher import MetricsPublisher
from libs.ops.metrics.metrics import Metrics
from libs.ops.sqs_clients import AsyncConsumer, AsyncProducer
from libs.ops.data_schema.data_input_schema import position_info_schema, input_logic_check
from libs.ops.data_schema.external_schema import ocr_external_result_schema
from libs.ops.data_schema.external_schema_init_pipeline import ocr_external_result_schema_init_pipeline


# from blueprints import *
from libs.ops.logging_ops import gen_logging_string
from libs.ops.preprocess_postprocess_ops import *
# from libs.util.struct_util import *
from libs.ops.pipeline_ops import process_blueprint_groups
from time import time, sleep
from copy import deepcopy
from version import version
from json.decoder import JSONDecodeError

import gc

# def quick_check_int64(layer):
#     if type(layer) == dict:
#         for key, val in layer.items():
#             quick_check_int64(key)
#             quick_check_int64(val)
            
#     elif type(layer) == list:
#         for nextlayer in layer:
#             quick_check_int64(nextlayer)
#     else:
#         if type(layer) == np.int64:
#             print('int64 number: ', layer)

class SuperWorker:
    def __init__(self, output_dir=PipelineConfig.RUNTIME_RESULT_DIR, run_id=True,
                 debug_level=-1, use_queue=False, return_zero=True, blueprint_group='@all',
                 nonlinear=False, is_inferred_by_cpu=False, download_ckpt_only=False):
        """
        OCR pipeline worker
        Args:
            output_dir: string
                directory for saving all model output, medium step and debug images
            debug_level: int
                debug level for determine
            use_queue: bool
                whether to use rabbit queue or not
            return_zero: bool
                whether continue running the pipeline when there's an exception
        """
        self.name = "**pipeline**"
        self.data = {}  # this is the dictionary to store all data, must keep this as a dict to keep expand compatibility
        self.blueprint_group = process_blueprint_groups()[blueprint_group]  # TODO try catch and logging
        self.supported_flags = []
        if blueprint_group[0] == '@':
            self.supported_flags += Blueprints.GROUPS[blueprint_group]
        else:
            self.supported_flags = [blueprint_group]
        self.output_dir = output_dir
        self.run_id = run_id
        self.is_inferred_by_cpu = is_inferred_by_cpu
        self.return_zero = return_zero  # for pytest and docker, always return success or not
        self.debug_level = debug_level
        self.strict_schema = True

        if debug_level > 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # --- API -----------------------------------------------------------------------------------------------------
        self.use_queue = use_queue  # use web based queue or not,
        self.input_topic = "feed-ocr"
        self.output_topic = "fetch-ocr"
        if self.debug_level >= 0 and (not self.output_dir):
            raise ValueError('output dir not specified')
        if use_queue:
            self.env = os.environ.get('ENV')
            self.amqp_producer = AsyncProducer()
            self.metrics_publisher = MetricsPublisher()
        # --- load models ----------------------------------------------------------------------------------------------
        self.modules = {}
        if download_ckpt_only:
            self.download_module_ckpts()
        else:
            self.load_modules()
    
    def download_module_ckpts(self):
        for blueprint_group_data in deepcopy(self.blueprint_group):
            module_name, flag = blueprint_group_data[0], blueprint_group_data[1]
            try:
                module = Blueprints.MODULES[module_name]
                for submodule in module:
                    submodule.download_ckpts(flag)
                logging.info(gen_logging_string(module_name, '', 'module_download_status', '<SUCCEED>'))
            except:
                logging.error(gen_logging_string(module_name, '', 'module_download_status', '<ERROR>'),
                              exc_info=True)
                raise RuntimeError("download {} module error".format(module_name))
        gc.collect()

        
    def load_modules(self):
        self.modules = {}
        for blueprint_group_data in deepcopy(self.blueprint_group):
            module_name, flag = blueprint_group_data[0], blueprint_group_data[1]
            try:
                module = Blueprints.MODULES[module_name]
                loaded_module = []
                for submodule in module:
                    submodule.load(flag)
                    if not self.is_inferred_by_cpu:
                        sleep(5) # avoid collision when load
                    loaded_module.append(submodule)
                self.modules[module_name] = loaded_module
                logging.debug(gen_logging_string(module_name, '', 'module_load_status', '<SUCCEED>'))

            except:
                logging.error(gen_logging_string(module_name, '', 'module_load_status', '<ERROR>'),
                              exc_info=True)
                raise RuntimeError("load {} module error".format(module_name))
        gc.collect()
        if not self.is_inferred_by_cpu:
            torch.cuda.empty_cache()
            max_gpu_memory = torch.cuda.max_memory_allocated()
            max_gpu_memory_readable = max_gpu_memory / (1024 ** 2)
            logging.info(gen_logging_string("idle", 'load', 'module_max_gpu_memory',
                                            '${:.3f}$MB'.format(max_gpu_memory_readable)))

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        if self.debug_level > 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def run_module(self, data, module_name, flags, output_dir):
        # TODO check input
        # TODO check output
        # TODO log  ·
        # TODO debug
        start_time = time()
        retry = 0
        max_retry = 3
        data_in = deepcopy(data)
        while retry < max_retry:
            data = deepcopy(data_in)
            with torch.inference_mode():
                try:
                    if not self.is_inferred_by_cpu and torch.cuda.is_available():
                        torch.cuda.reset_max_memory_allocated()
                    for submodule in self.modules[module_name]:
                        if self.debug_level > 0:
                            if not (os.path.exists(join(output_dir, submodule.name))):
                                os.makedirs(join(output_dir, submodule.name))
                        submodule.set_debug_info(self.debug_level, join(output_dir, submodule.name))
                        data = submodule.run(data, flags)
                    if not self.is_inferred_by_cpu:
                        max_gpu_memory = torch.cuda.max_memory_allocated()
                        max_gpu_memory_readable = max_gpu_memory / (1024 ** 2)
                        logging.info(gen_logging_string(module_name, data['id'], 'module_max_gpu_memory',
                                                        '${:.3f}$MB'.format(max_gpu_memory_readable)))
                    logging.debug(gen_logging_string(module_name, data['id'], 'module_runtime_status', '<SUCCEED>'))
                    logging.info(gen_logging_string(module_name, data['id'], 'module_time_spent',
                                                    '${:.3f}$s'.format(time() - start_time)))

                    # gc.collect()
                    if not self.is_inferred_by_cpu:
                        torch.cuda.empty_cache()

                    return data

                except Exception as e:

                    if 'CUDA' in str(e):
                        """
                        OOM: reload moudules and try again, if tryed 3 times or OOM when reload, it won't
                        retry again
                        """
                        sleep(5)
                        retry += 1
                        # reload modules
                        del self.modules
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.modules = {}
                        self.load_modules()
                        logging.info(gen_logging_string(module_name, data['id'], 'module_runtime_status',
                                                        '<RETRYED_OOM>, {}'.format(retry)))
                    else:
                        logging.error(gen_logging_string(module_name, data['id'], 'module_runtime_status', '<FAILED>'),
                                    exc_info=True)
                        logging.info(
                            gen_logging_string('PIPELINE', data['id'], 'pipeline_runtime_status', '${}$'.format("FAILED")))
                        raise RuntimeError("RUN {} failed".format(module_name))
        logging.error(gen_logging_string(module_name, data['id'], 'module_runtime_status', '<FAILED>'),
                      exc_info=True)
        logging.info(
            gen_logging_string('PIPELINE', data['id'], 'pipeline_runtime_status', '${}$'.format("FAILED")))
        raise RuntimeError("RUN {} failed".format(module_name))

    def run(self, data, flag):
        if self.run_id:
            run_id = datetime.datetime.today().isoformat()
            run_id = re.sub(r'[-:.]', '_', run_id)
            save_dir = join(self.output_dir, "{}{}".format(data['id'], "__" + run_id))
        else:
            save_dir = join(self.output_dir, "{}".format(data['id'].split('@')[0]))

        if self.debug_level >= 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if flag[0] == '_':  # preprocessed [deprecated]
            blueprint = deepcopy(Blueprints.BLUPRINTS[flag[1:]])
            blueprint[0] = (blueprint[0][0], ['skip'])
        elif flag[0] == '*':  # preprocess only
            blueprint = deepcopy(Blueprints.BLUPRINTS[flag[1:]])
            blueprint = [blueprint[0]]
        else:
            if flag not in self.supported_flags:
                raise RuntimeError("{} is not currently in {} group".format(flag, self.supported_flags))
            blueprint = deepcopy(Blueprints.BLUPRINTS[flag])

        logging.info(gen_logging_string('PIPELINE', data['id'], 'pipeline_flag_used', '${}$'.format(str(flag))))
        time_start = time()

        if self.debug_level == -10:
            os.makedirs(save_dir, exist_ok=True)
            # raw_input.json
            data_input_debug = deepcopy(data)
            data_input_debug['info']['image']['orig'] = 'input_img.jpg'
            data_input_debug['info']['image']['orig_color'] = 'input_img.jpg'
            with open(join(save_dir, 'raw_input.json'), 'w') as fp:
                json.dump(data_input_debug, fp, sort_keys=True, indent=4, ensure_ascii=False)

        # save raw input
        if self.debug_level >= 1:
            data_input_debug = deepcopy(data)
            img = data_input_debug['info']['image']['orig']
            cv2.imwrite(join(save_dir, 'input_img.jpg'), img) # lossy, change to png to lossless
            data_input_debug['info']['image']['orig'] = 'input_img.jpg'
            data_input_debug['info']['image']['orig_color'] = 'input_img.jpg'
            with open(join(save_dir, 'raw_input.json'), 'w') as fp:
                json.dump(data_input_debug, fp, sort_keys=True, indent=4, ensure_ascii=False)

        if re.match('one_eb_rg2_.*_align.*', flag) and self.strict_schema:
            try:
                position_info_schema.validate(data['info']['question_position_info'])
            except Exception as e:
                raise ValueError(f'input data is invalidate! details: {e}')
            try:
                input_logic_check(data['info']['meta'], data['info']['question_position_info'])
            except Exception as e:
                # raise ValueError(f'input data is logic invalidate! details: {e}')    
                logging.warn(f'input data is logic invalidate! details: {e}')    
            
        for module_name, flags in blueprint:
            data = self.run_module(data, module_name, flags, save_dir)
        logging.debug(gen_logging_string('PIPELINE', data['id'], 'pipeline_runtime_status', '${}$'.format("SUCCEED")))
        logging.info(
            gen_logging_string('PIPELINE', data['id'], 'pipeline_time_spent', '${:.3f}$s'.format(time() - time_start)))
        if 'rg2' in flag:
            if 'init' in flag:
                data['info']['v2_struct'] = ocr_api2_internal_to_external(data['info']['struct'],data['info']['image']['orig_size'][0],data['info']['image']['orig_size'][1], is_init_pipeline=True)
                if self.strict_schema:
                    ocr_external_result_schema_init_pipeline.validate(data['info']['v2_struct'])
            else:
                data['info']['v2_struct'] = ocr_api2_internal_to_external(data['info']['struct'],data['info']['image']['orig_size'][0],data['info']['image']['orig_size'][1], is_init_pipeline=False)
                if self.strict_schema:
                    ocr_external_result_schema.validate(data['info']['v2_struct'])
        
        if self.debug_level >= 1 and False:
            if 'struct' in data['info'] and not ('new_format' in data['info'] and data['info']['new_format']):
                # save debug image
                debug_img = visualize_struct(data['info']['image']['orig'], data['info']['struct'])
                cv2.imwrite(join(save_dir, 'debug.jpg'), debug_img, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])  # debug image
                del debug_img
                # debug_json = posprocess_web_data(deepcopy(data))
                # save encoded result
                result_json_postprocessed = dummy_posprocess_web_data(deepcopy(data), flag)
                with open(join(save_dir, 'result_postprocessed.json'), 'w') as fp:
                    json.dump(result_json_postprocessed['struct'], fp, sort_keys=True, indent=4, ensure_ascii=False)
        if self.debug_level >= 0 and False:
            if 'struct' in data['info']:
                result_json = encode_struct(deepcopy(data['info']['struct']))
                with open(join(save_dir, 'result.json'), 'w') as fp:
                    # quick_check_int64(result_json)
                    # print(result_json)
                    json.dump(result_json, fp, sort_keys=True, indent=4, ensure_ascii=False)
                # save readable result
                result_json_readable = struct_to_readable(deepcopy(data['info']['struct']))
                with open(join(save_dir, 'result_readable.json'), 'w') as fp:
                    fp.write(result_json_readable)
                logging.info(
                    gen_logging_string('PIPELINE', data['id'], 'pipeline_debug_save_dir', '${}$'.format(str(save_dir))))
                del result_json
            if 'v2_struct' in data['info']:
                result_json = data['info']['v2_struct']
                with open(join(save_dir, 'v2_result.json'), 'w') as fp:
                    json.dump(result_json, fp, sort_keys=True, indent=4, ensure_ascii=False)
                v2_external_debug_img = visualize_v2_external_struct(cv2.cvtColor(data['info']['image']['orig'].copy(), cv2.COLOR_GRAY2BGR), result_json)
                cv2.imwrite(join(save_dir, 'v2_debug.jpg'), v2_external_debug_img, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])  # debug image
                v2_e2e_debug_img = visuzliaze_v2_struct_e2e(cv2.cvtColor(data['info']['image']['orig'].copy(), cv2.COLOR_GRAY2BGR), result_json)
                cv2.imwrite(join(save_dir, 'v2_strcut_e2e.jpg'), v2_e2e_debug_img, [cv2.IMWRITE_JPEG_OPTIMIZE, int(True), cv2.IMWRITE_JPEG_QUALITY, 70])
    
        if self.debug_level == -10 and False:
            os.makedirs(save_dir, exist_ok=True)

            # result.json
            if 'struct' in data['info']:
                result_json = encode_struct(deepcopy(data['info']['struct']))
                result_json['orig_size'] = data['info']['image']['orig_size']
                with open(join(save_dir, 'result.json'), 'w') as fp:
                    json.dump(result_json, fp, sort_keys=True, indent=4, ensure_ascii=False)

            # v2_result.json
            if 'v2_struct' in data['info']:
                result_json = data['info']['v2_struct']
                result_json['orig_size'] = data['info']['image']['orig_size']
                with open(join(save_dir, 'v2_result.json'), 'w') as fp:
                    json.dump(result_json, fp, sort_keys=True, indent=4, ensure_ascii=False)

        gc.collect()
        if not self.is_inferred_by_cpu:
            torch.cuda.empty_cache()

        return data

    async def start_process(self, raw_input):
        # logging.info(f'[debug] input data: {raw_input}')
        try:
            data_in, flag = await preprocess_web_data(raw_input, self.env)
            start_model = time()
            data = self.run(data_in, flag)
            Metrics.model_duration.add_duration(time() - start_model)
            # post process
            raw_output = await posprocess_web_data(data, flag)
        except ValueError as e:
            Metrics.download_image_fail.add_count()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            status_message = f"{raw_input['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))
            logging.error(status_message)

            if 'subject' not in raw_input:
                raw_output = {'id': raw_input['img_key'],
                              'image': {},
                              'struct': None,
                              'status': 4,
                              'status_message': status_message
                              }
            else:
                raw_output = {'imageId': raw_input['img_key'],
                              'page': None,
                              'status': 4,
                              'status_message': status_message
                              }
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            status_message = f"{raw_input['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))
            logging.error(status_message)
            if 'subject' not in raw_input:
                raw_output = {'id': raw_input['img_key'],
                              'image': {},
                              'struct': None,
                              'status': 4,
                              'status_message': status_message
                              }
            else:
                raw_output = {'imageId': raw_input['img_key'],
                              'page': None,
                              'status': 4,
                              'status_message': status_message
                              }
            Metrics.unexpected_error.add_count()
        raw_output['version'] = version
        if 'timestamp' in raw_input:
            raw_output['timestamp'] = raw_input['timestamp']
        if 'flags' in raw_input:
            raw_output['flags'] = raw_input['flags']
        if 'subject' not in raw_input:
            raw_output['blueprint_flag'] = raw_input.get('blueprint_flag', '')
        if not self.return_zero:
            sys.exit("{} process failed".format(raw_input['img_key']))
        return raw_output

    async def process_message(self, message, output_queue, priority_index=None):
        try:
            Metrics.receive_count.add_count()

            raw_input = json.loads(message["Body"])
            raw_output = await self.start_process(raw_input)
            count = 0
            while count <= PipelineConfig.QUEUE_CONN_RETRY_LIMIT:
                try:
                    await self.amqp_producer.publish(raw_output, output_queue)
                    break
                except:
                    mq_exc_info = sys.exc_info()
                    logging.info(f'{raw_input["img_key"]} retry to publish ocr results...')
                    await asyncio.sleep(1)
                    count += 1
            if count > PipelineConfig.QUEUE_CONN_RETRY_LIMIT:
                raise mq_exc_info[1]
            else:
                if time() - raw_output['timestamp'] > PipelineConfig.MAX_SUCCESS_DURATION:
                    Metrics.fail_count.add_count()
                logging.info(f'{raw_input["img_key"]} send ocr result back to queue {output_queue} successfully')
            Metrics.ocr_pipeline_duration.add_duration(time() - raw_output['timestamp'])
        except JSONDecodeError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.error(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            Metrics.unexpected_error.add_count()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.error(f"{raw_input['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            Metrics.unexpected_error.add_count()
        self.metrics_publisher.publish_metrics(Metrics.metrics, priority_index)

    def run_forever(self):
        # publish_metrics_task = [metrics_publisher.publish_periodic, metrics]
        AsyncConsumer(self)
