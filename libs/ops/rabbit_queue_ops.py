import asyncio
import aio_pika
import logging
import ssl

from credentials import get_password
from pipeline_config_v2 import RABBIT_MQ_ENV_CONFIG


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class AsyncConsumer:
    def __init__(self, superworker, publish_metrics_task):
        self.ip = superworker.ip
        # for OCR API
        self.queue_name = superworker.input_topic
        self.process_message = superworker.process_message
        # for MQ Mode
        self.queue_name_430 = superworker.input_topic + '_430'
        self.process_message_430 = superworker.process_message_430
        self.superworker = superworker
        self.mq_config = RABBIT_MQ_ENV_CONFIG[self.superworker.env]

      
        if self.superworker.env == 'preprod-dev':
            self.queue_name_qa = self.queue_name + '_qa'
            self.process_message_qa = superworker.process_message_qa
        
        self.publish_metrics_task = publish_metrics_task
        loop = asyncio.get_event_loop()
        self.superworker.loop = loop
        loop.set_exception_handler(self.handle_exception)
        try:
            loop.run_until_complete(self.start_all(loop))
            loop.run_forever()
        except Exception as e:
            logging.error(str(e), exc_info=True)
            loop.stop()
            logging.info("Stopped to process messages")

    async def start_all(self, loop):
        task_func = self.publish_metrics_task.pop(0)
        task_args = self.publish_metrics_task
        task = loop.create_task(task_func(*task_args))
        await asyncio.gather(self.main(loop), task)

    def handle_exception(self, loop, context):
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get("exception", context["message"])
        logging.error(f"Caught exception: {msg}")

    async def main(self, loop):
        if self.superworker.is_inferred_by_cpu:
            host = f"{self.mq_config['broker_id']}.mq.{self.mq_config['region']}.amazonaws.com"
            if self.mq_config['region'].startswith('cn'):
                host += '.cn'
            consumer_connection = await aio_pika.connect_robust(
                login=self.mq_config['username'], password=get_password(self.superworker.env), host=host, port=5671,
                loop=loop, timeout=10, reconnect_interval=1, heartbeat=self.mq_config['heartbeat'], virtualhost=self.mq_config['virtual_host'], ssl=True,
                ssl_options=dict(
                    ca_certs="libs/ops/AmazonRootCA1.pem",
                    cert_reqs=ssl.CERT_REQUIRED,
            ))
        else:
            consumer_connection = await aio_pika.connect_robust(
                login='ocr_pipeline', password=get_password(self.superworker.env), host=self.ip, port=5672,
                loop=loop,
                timeout=10,
                virtualhost=self.mq_config['virtual_host'],
                heartbeat=self.mq_config['heartbeat']
            )
        consumer_channel = await consumer_connection.channel()
        # prefetch greater than 1 will cause CUDA ERROR in async threading mode
        await consumer_channel.set_qos(prefetch_count=1)

        # Declaring queue
        queue = await consumer_channel.declare_queue(
            self.queue_name, durable=True
        )
        logging.info(f'start consuming queue {self.queue_name}...')
        await queue.consume(self.process_message)

        queue_430 = await consumer_channel.declare_queue(
            self.queue_name_430, durable=True
        )
        logging.info(f'start consuming queue {self.queue_name_430}...')
        await queue_430.consume(self.process_message_430)
        
        if self.superworker.env == 'preprod-dev':
            queue_qa = await consumer_channel.declare_queue(
                self.queue_name_qa, durable=True
            )
            logging.info(f'start consuming queue {self.queue_name_qa}...')
            await queue_qa.consume(self.process_message_qa)
