from blueprints import Blueprints
from libs.util.local_rabbitmq_util import *
from libs.ops.logging_ops import gen_logging_string
import torch
from time import time


class NodeWorker:
    # wrapper for pipeline node, only need a main function to start this
    # the serialize part, everything in run func in 
    def __init__(self,blueprint_name, node_name):
        """get parameters ready
        """
        self.node_name = node_name
        self.blueprint_name = blueprint_name
        self.node_name = node_name
        self.sub_blueprint,self.dst_node_name = Blueprints.BLUPRINTS[blueprint_name][node_name]
        self.modules = []
        self.queue_worker = SimpleQueueWorker("@{}@{}".format(self.blueprint_name,
                                                              self.node_name),
                                              "@{}@{}".format(self.blueprint_name,
                                                              self.dst_node_name),
                                              self.run)
    def load(self):
        """load every module in this Node and store it in self.modules
        """
        for module_info in self.sub_blueprint:
            module_name, flags = module_info
            for submodule in Blueprints.MODULES[module_name]:
                try:
                    # load every module in this node
                    submodule.load(flags)# todo flags and stuff
                    self.modules.append((submodule,flags)) #TODO multiple flags
                    logging.info(gen_logging_string(module_name, '', 'module_load_status', '<SUCCEED>'))

                except:
                    logging.error(gen_logging_string(module_name, '', 'module_load_status', '<ERROR>'),
                              exc_info=True)
                    raise RuntimeError("load {} module error".format(module_name))
                sleep(5)
        gc.collect()
        torch.cuda.empty_cache()
    
    def run(self,message):
        """run one node (part of pipeline) on serialized message

        Args:
            message (bytes): serialized input data

        Returns:
            [bytes]: serialized input data
        """
        # TODO add debug and stuff
        node_time_start = time()
        t = time()
        data = deserialize_pipeline_data(message)
        if 'status' not in data: # compatiablity
            data['status'] = 'SUCCEED'
            logging.info(gen_logging_string("data_guardian", data['id'], 'data_note',
                         '${}$_missing_in_data'.format('status')))

        logging.info(gen_logging_string("deserialize-{}".format(self.node_name), data['id'], 'code_time_spent',
                                '${:.3f}$s'.format(time() - t)))
        if data['status'] == 'SUCCEED':

            with torch.inference_mode():
                try:
                    for module,flags in self.modules:
                        start_time = time()
                        data = module.run(data,flags) # todo use data info
                        module_name = module.name

                        logging.info(gen_logging_string(module_name, data['id'], 'module_runtime_status', '<SUCCEED>'))
                        logging.info(gen_logging_string(module_name, data['id'], 'module_time_spent',
                                                                '${:.3f}$s'.format(time() - start_time)))

                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception as e:
                    data['status'] == 'FAILED'
                    logging.error(gen_logging_string(module_name, data['id'], 'module_runtime_status', '<FAILED>'),
                                exc_info=True)
                    logging.info(
                        gen_logging_string('PIPELINE', data['id'], 'pipeline_runtime_status', '${}$'.format("FAILED")))
            
        t = time()
        message = serialize_pipeline_data(data)
        logging.info(gen_logging_string("serialize-{}".format(self.node_name), data['id'], 'code_time_spent',
                                '${:.3f}$s'.format(time() - t)))
        logging.info(gen_logging_string(
            "@{}@{}".format(self.blueprint_name,self.node_name),
            data['id'],
            'module_time_spent',
            '${:.3f}$s'.format(time() - node_time_start)))
        return message
    
    def run_forever(self):
        """run the worker
        """
        # TODO post to multiple branches
        self.queue_worker.consume_forever()
    # also need a blueprint checker/parser



