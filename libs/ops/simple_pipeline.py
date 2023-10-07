from glob import glob
import os
from os.path import join, basename, dirname, exists
from libs.super_api_v2 import SuperWorker
import cv2
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import pika
import json
from time import sleep, time
import base64
import numpy as np
import logging
import sys
from random import random
from libs.ops.logging_ops import gen_logging_string
from pipeline_config_v2 import PipelineConfig
import gc
import datetime
import aio_pika
from aio_pika import IncomingMessage
# from libs.util.image_util import cap_longest_edge
# from libs.util.local_rabbitmq_util import deserialize_pipeline_data
# from libs.util.cpu_inference_util import set_cpu_affinity,get_cpu_affinity
# from libs.ops.preprocess_postprocess_ops import get_exercise_book_questions_info, preprocess_web_data
# from libs.util.onnx_util import get_inference_threads
from tqdm import tqdm
import asyncio
import torch
import multiprocessing as mp
import warnings
import traceback

def dispatch_payloads(num_cpus, workers, payload_idxes):
    if len(payload_idxes) <= 0:
        return []
    dispatch_params = []
    if len(payload_idxes) <= workers:
        workers = len(payload_idxes)
    payloads_per_worker = len(payload_idxes)//workers
    assert payloads_per_worker >= 1
    hard_worker_count = len(payload_idxes) - payloads_per_worker*workers
    for worker_idx in range(workers):
        beg = worker_idx * payloads_per_worker
        end = (worker_idx + 1) * payloads_per_worker        
        batch_payload_idxes = payload_idxes[beg: end]
        if worker_idx < hard_worker_count:
            tail_beg = len(payload_idxes) - hard_worker_count + worker_idx
            batch_payload_idxes += payload_idxes[tail_beg: tail_beg+1]
        cpu_ids = [(worker_idx*get_inference_threads()+i) % num_cpus for i in range(get_inference_threads())]
        dispatch_params.append((cpu_ids, batch_payload_idxes))
    return dispatch_params

class SimplePipeline:
    """
    simple and lazy ocr pipeline wrapper for e2e inference tests
    if you want more efficient, more complicated runs, please use comsumer/publisher
    Learnable, INC, All rights researved.
    """
    def __init__(self, output_dir, blueprint, debug_level=1, run_id=False, n_proc=1,
                 skip_failed=False, blueprint_group = ""):
        """
        init session, load model, init pipeline,
        by design, every run session can only run one blueprint

        Args:
            output_dir (str): root of the directory to store debug outputs
            blueprint (str): blueprint for this session
            debug_level (int, optional): debug level:   -1 for no output,
                                                        0 for only final json.
                                                        1 for complete debug.
                                                        Defaults to 1.
            run_id (bool, optional): generate a run id for this run to avoid overwrite result. Defaults to False.
            n_proc (int, optional): number of procesess, Defaults to 1. GPU multiprocess is unstable and not recommended.
            skip_failed (bool, optional): throw error or not if encouters. Defaults to False. #TODO reimplement this
        """
        is_inferred_by_cpu = os.environ.get('OCR_HARDWARE_ENV', '') == 'cpu'
        if len(blueprint_group) > 0:
            self.pipeline = SuperWorker(output_dir=output_dir, debug_level=debug_level, run_id=run_id,
                                    blueprint_group=blueprint_group, is_inferred_by_cpu=is_inferred_by_cpu)
        else:
            self.pipeline = SuperWorker(output_dir=output_dir, debug_level=debug_level, run_id=run_id,
                                        blueprint_group=blueprint, is_inferred_by_cpu=is_inferred_by_cpu)
        self.blueprint = blueprint
        self.n_proc = n_proc
        self.skip_failed = skip_failed

    async def run_raw(self,raw_data):
        """run pipeline using raw request data

        Args:
            raw_data (dict): raw request data in dict format
        """
        try:
            # preprocess raw data
            internal_input_data, _ = await preprocess_web_data(raw_data, 'dev')
            self.pipeline.run(internal_input_data, self.blueprint)
        except ValueError as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.warn(f"{raw_data['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.error(f"{raw_data['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    
    def run_raw_worker(self,cpu_id,raw_requests):
        """run pipeline(a worker) on a list of images. Currently, in use only when
        self.n_proc>1

        Args:
            cpu_id (list): list of cpu core to run the worker on
            raw_requests (list of string): list of image paths waiting to run
        """
        set_cpu_affinity(cpu_id)
        start =  time()
        num_threads = 1
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)
        interp_threads = get_inference_threads()
        if torch.get_num_interop_threads() != interp_threads:
            torch.set_num_interop_threads(interp_threads)
        p = mp.current_process()
        warnings.filterwarnings('ignore')

        logging.info(f'Starting: {p.name}, {p.pid}, on core: {os.sched_getaffinity(p.pid)}, num images: {len(raw_requests)}')
        for raw_request in tqdm(raw_requests):
            try:
                asyncio.run(self.run_raw(raw_request))
            except Exception:
                logging.error("RUN FOLDED FAILED CASE: " + raw_request['img_key'])
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logging.error(f"{raw_request['img_key']} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        logging.info(f'Complete process: {p.name} in {time()-start} seconds')
        p.close()

    def run_raw_session(self, raw_requests):
        if self.n_proc == 1:
            self.run_raw_worker([i for i in range(min(os.cpu_count(), get_inference_threads()))], raw_requests)
        else:
            jobs = []
            dispatch_params = dispatch_payloads(os.cpu_count(), self.n_proc, [i for i in range(len(raw_requests))])            
            for cpu_ids, payload_idxes in dispatch_params:
                batch_raw_requests = [raw_requests[id] for id in payload_idxes]
                p = mp.Process(target=self.run_raw_worker,args=(cpu_ids,batch_raw_requests,))
                jobs.append(p)
                p.start()
            for job in jobs:
                job.join()

    def run(self, img_path, extra_info, id_tag=""):

        """run pipeline on a single image

        Args:
            img_path (str): path to the image
            extra_info (dict): dict of extra info, all keys in the dictit'll be directly copy to the input['info'] dict
        """
        # get image id from whole path for easy locate failed image
        id = "^".join(img_path.replace('.','_').split('/'))
        try:
            # load image, # TODO url
            img = cv2.imread(img_path)
            id = id + "_" + id_tag
            # inital input data
            data = {'id': id,
                    'format': 'input',
                    'info': {'image': {'orig': img}}
                    }
            # extra input data

            for key, value in extra_info.items():  ## extra dict are all assumed to be in info.
                data['info'][key] = value
            # run pipeline on the image, no result as it'll be saved in debug folder
            self.pipeline.run(data, self.blueprint)
        except ValueError as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.warn(f"{id} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.error(f"{id} " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))

    def run_worker(self,cpu_id,img_paths,extra_info_list,id_tag):
        """run pipeline(a worker) on a list of images. Currently, in use only when
        self.n_proc>1

        Args:
            cpu_id (list): list of cpu core to run the worker on
            img_paths (list of string): list of image paths waiting to run
            extra_info_list (list of dict): list of extra info for each image, see run() for more detail
        """
        set_cpu_affinity(cpu_id)
        start =  time()
        num_threads = 1
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)
        interp_threads = get_inference_threads()
        if torch.get_num_interop_threads() != interp_threads:
            torch.set_num_interop_threads(interp_threads)
        p = mp.current_process()
        warnings.filterwarnings('ignore')

        logging.info(f'Starting: {p.name}, {p.pid}, on core: {os.sched_getaffinity(p.pid)}, num images: {len(img_paths)}')
        for img_idx,img_path in tqdm(enumerate(img_paths)):
            self.run(img_path, extra_info_list[img_idx],id_tag)
        logging.info(f'Complete process: {p.name} in {time()-start} seconds')
        p.close()

    def run_session(self, img_paths, extra_info_list = None, id_tag=""):

        """run pipeline on a list of images

        Args:
            img_paths (list of string): list of image paths waiting to run
            meta_data (list of dict): list of extra info for each image, see run() for more detail
        """
        if extra_info_list is None:
            extra_info_list = [{} for _ in img_paths]
        if len(img_paths) != len(extra_info_list):
            raise ValueError("img_paths and meta_data must have same length")

        if self.n_proc == 1:
            self.run_worker([i for i in range(min(os.cpu_count(), get_inference_threads()))], img_paths, extra_info_list, id_tag)
        else:
            jobs = []
            dispatch_params = dispatch_payloads(os.cpu_count(), self.n_proc, [i for i in range(len(extra_info_list))])            
            for cpu_ids, payload_idxes in dispatch_params:
                worker_img_paths = [img_paths[id] for id in payload_idxes]
                worker_extra_info_list = [extra_info_list[id] for id in payload_idxes]
                p = mp.Process(target=self.run_worker,args=(cpu_ids,worker_img_paths, worker_extra_info_list, id_tag, ))
                jobs.append(p)
                p.start()
            for job in jobs:
                job.join()

            for job in jobs:
                job.join()

class SimplePipelineConsumer:
    # main stuff
    def __init__(self, debug_level=1, run_id=False, skip=True, blueprint_group='@all',nonlinear=False):
        # pipeline

        self.nonlinear = nonlinear
        if nonlinear:
            self.pipeline = SuperWorker(debug_level=debug_level, run_id=run_id, blueprint_group=blueprint_group,nonlinear=True)
        else:
            self.pipeline = SuperWorker(debug_level=debug_level, run_id=run_id, blueprint_group=blueprint_group)
        self.skip = skip  # skip all input data, work as a queue clearer
        # # connections
        # credentials = pika.PlainCredentials('guest', 'guest')
        # parameters = pika.ConnectionParameters('localhost', heartbeat=0)
        self.nonlinear = nonlinear
        # queues
        self.in_queue_name = "pipeline_local_{}".format(debug_level)  # incoming queue
        # self.connection = pika.BlockingConnection(parameters)
        # self.channel = self.connection.channel()
        # self.channel.exchange_declare(exchange="test_exchange", exchange_type="direct", passive=False,
        #                               durable=True, auto_delete=False)
        # self.channel.queue_declare(queue=self.in_queue_name)
        # self.channel.queue_bind(queue=self.in_queue_name, exchange="test_exchange", routing_key="standard_key")

        self.out_queue_name = "pipeline_local_result"
        self.out_channel = None
        # self.out_connection = pika.BlockingConnection(parameters)
        # self.out_channel = self.out_connection.channel()
        # self.out_channel.exchange_declare(exchange="test_exchange", exchange_type="direct", passive=False,
        #                                   durable=True, auto_delete=False)
        # self.out_channel.queue_declare(queue=self.out_queue_name)
        # self.out_channel.queue_bind(queue=self.out_queue_name, exchange="test_exchange", routing_key="standard_key")

    async def main(self, loop):
        in_connection = await aio_pika.connect_robust(
            "amqp://guest:guest@localhost?heartbeat=0",
            loop=loop
        )

        in_channel = await in_connection.channel()
        await in_channel.set_qos(prefetch_count=1)

        in_queue = await in_channel.declare_queue(
            self.in_queue_name, durable=False
        )

        out_connection = await aio_pika.connect_robust(
            "amqp://guest:guest@localhost?heartbeat=0",
            loop=loop
        )

        self.out_channel = await out_connection.channel()
        await self.out_channel.set_qos(prefetch_count=1)

        await self.out_channel.declare_queue(
            self.out_queue_name, durable=False
        )

        await in_queue.consume(self.do_work)

    # @staticmethod
    # def ack_message(channel, delivery_tag):
    #     """Note that `channel` must be the same pika channel instance via which
    #     the message being ACKed was retrieved (AMQP protocol constraint).
    #     """
    #     if channel.is_open:
    #         channel.basic_ack(delivery_tag)
    #     else:
    #         # Channel is already closed, so we can't ACK this message;
    #         # log and/or do something that makes sense for your app in this case.
    #         pass

    async def do_work(self, message: IncomingMessage):
        async with message.process():
            try:
                ###### work ######
                t = time()
                data = json.loads(message.body)
                img = cv2.imread(data['info']['image']['file_path'],cv2.IMREAD_GRAYSCALE)
                orig_h, orig_w = img.shape[:2]
                data['info']['image']['orig'] = img
                img_folder = dirname(data['info']['image']['file_path'])
                if img_folder[-1] == "/":
                    img_folder = img_folder[:-1]

                result_folder = join(dirname(img_folder), "{}_PipelineResult/".format(basename(img_folder)))
                if data['result_dir'] is not None:
                    result_folder = data['result_dir']
                # print(result_folder)
                # sleep(3)
                self.pipeline.set_output_dir(result_folder)
                if self.nonlinear:
                    self.pipeline.publish_to_nonlinear_queue(data, data['flag'])
                else:
                    self.pipeline.run(data, data['flag'])
                    logging.info(gen_logging_string('=LOCAL_CONSUME=', data['id'], 'consumer_time',
                                                    '${:.3f}$s'.format(time() - t)))

                    ######
                    ocr_message = {'id': data['id'],
                                'status': 'SUCCEED'}

                    await self.out_channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(ocr_message).encode(),
                            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT
                        ),
                        routing_key=self.out_queue_name
                    )
                    del data
                    del img
                    gc.collect()
            except KeyboardInterrupt:
                logging.error('Interrupted')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
            except:
                logging.error("{:<30}[{}] [{}] in {:.3f}s".format('[=LOCAL_CONSUME=]',
                                                                    data['id'],
                                                                    'FAILED', time() - t), exc_info=True)
                ocr_message = {'id': data['id'],
                            'status': 'FAILED'}
                await self.out_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(ocr_message).encode(),
                        delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT
                    ),
                    routing_key=self.out_queue_name
                )


class SimplePipelinePublisherTraced:
    """
    a amqp publisher that publishes data into "pipeline_local" queue to be waited to process
    """

    def __init__(self, avoid_collide=True, debug_level=1, allow_fail = False, result_queue_name = "pipeline_local_result"):
        self.avoid_collide = avoid_collide
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=0, ), )
        self.channel = self.connection.channel()
        self.work_queue_name = "pipeline_local_{}".format(debug_level)
        self.channel.queue_declare(queue=self.work_queue_name)

        self.result_connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=0, ), )
        self.result_channel = self.result_connection.channel()
        self.result_queue_name = result_queue_name
        self.result_channel.queue_declare(queue=self.result_queue_name)

        self.published_ids = []
        self.succeed_ids = []
        self.failed_ids = []
        self.total_imgs = 0
        self.session_id = datetime.datetime.today().isoformat()
        logging.info(gen_logging_string('=LOCAL_PUBLISHER=', self.session_id, 'session_status',
                                        '{STARTED}'))
        self.start_time = time()
        self.time_stamps = [time()]
        self.allow_fail = allow_fail

    def simple_publish(self, message):
        queue_name = self.work_queue_name
        # self.channel.queue_declare(queue=queue_name)
        self.channel.basic_publish(exchange='',
                                   routing_key=queue_name,
                                   body=json.dumps(message))

    def publish_images(self, img_fname, flag, result_dir=None, tag='', extra_dict={}):
        img_id = '{}#{}@{}'.format(basename(img_fname),
                                   tag,
                                   ('%06x' % int(time() * 10000000)).upper()[-6:])
        data = {'id': img_id,
                'format': 'input',
                'info': {'image': {'orig': None,
                                   'file_path': img_fname}},
                'flag': flag,
                'result_dir': result_dir
                }
        for key, value in extra_dict.items(): ## extra dict are all assumed to be in info.
            data['info'][key] = value
        # print(datcom  2)
        self.simple_publish(data)
        self.published_ids.append(img_id)
        logging.info(gen_logging_string('=LOCAL_PUBLISHER=', img_id, 'queue_status', '{PUBLISHED}'))

    def trace_callback(self, ch, method, properties, body):
        if '^' in self.result_queue_name:
            img_id = self.published_ids[0] # todo real trace and error logging 
            status = "SUCCEED"
        else:
            message_dict = json.loads(body.decode())
            img_id = message_dict['id']
            status = message_dict['status']
        if img_id in self.published_ids:
            self.published_ids.remove(img_id)
            if status == "SUCCEED":
                self.succeed_ids.append(img_id)
            else:
                self.failed_ids.append(img_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            total_time = time() - self.start_time
            avg_time = total_time/max(len(self.succeed_ids),1)
            self.time_stamps.append(time())
            try:
                running_avg = (self.time_stamps[-1]-self.time_stamps[-11])/10
            except IndexError:
                running_avg = np.nan

            print('{} REMAIN, S: {}, F: {}, TBD: {:.2f}s, AVG: {:.2f}s/img'.format(
                len(self.published_ids),
                len(self.succeed_ids),
                len(self.failed_ids),
                running_avg*len(self.published_ids),
                running_avg,
            ),end='\r')

        if len(self.published_ids) == 0:
            print("[test session] => ALL SUCCEED")
            total_time = time() - self.start_time
            self.result_channel.stop_consuming()
            logging.info(gen_logging_string('=LOCAL_PUBLISHER=', self.session_id, 'session_status',
                                            '{FINISHED}'))
            logging.info(gen_logging_string('=LOCAL_PUBLISHER=', self.session_id, 'session_throughput_info',
                                            'Finished {} imgs, in {}s, avg {:.3f}s per img.'.format(
                                                self.total_imgs,
                                                total_time,
                                                total_time / max(self.total_imgs,1))))
            # sys.exit()
        if not self.allow_fail:
            if len(self.failed_ids) > 0:
                logging.error("ERROR {} images failed: {}".format(len(self.failed_ids), self.failed_ids))
                os.system("cat {} | grep ERROR -A 40".format(PipelineConfig.LOG_DIR))
                sys.exit(1)
                # TODO add log

        else:
            # print("an image from other session is finished")
            pass

    def listen_results(self):
        self.result_channel.basic_consume(queue=self.result_queue_name,
                                          auto_ack=False,
                                          on_message_callback=self.trace_callback)
        print('{} REMAIN, S: {}, F: {}, TBD: NaNs, AVG: NaNs/img'.format(
            len(self.published_ids),
            len(self.succeed_ids),
            len(self.failed_ids),

        ),end='\r')
        self.total_imgs = len(self.published_ids)
        self.result_channel.start_consuming()
