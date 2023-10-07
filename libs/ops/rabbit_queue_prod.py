import aio_pika
import os
import json
import logging
import ssl
from credentials import get_password
from pipeline_config_v2 import RABBIT_MQ_ENV_CONFIG

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class AsyncProducer:
    def __init__(self, supported_queue_names, env):
        self.supported_queue_names = supported_queue_names
        self.channel = None
        self.connection = None
        self.mq_password = get_password(env)
        self.mq_config = RABBIT_MQ_ENV_CONFIG[env]

    async def init(self, is_inferred_by_cpu, loop):
        
        if is_inferred_by_cpu:
            host = f"{self.mq_config['broker_id']}.mq.{self.mq_config['region']}.amazonaws.com"
            if self.mq_config['region'].startswith('cn'):
                host += '.cn'
            self.connection = await aio_pika.connect_robust(
                login=self.mq_config['username'], password=self.mq_password, host=host, port=5671,
                loop=loop, timeout=10, reconnect_interval=1, heartbeat=self.mq_config['heartbeat'], virtualhost=self.mq_config['virtual_host'], ssl=True,
                ssl_options=dict(
                    ca_certs="libs/ops/AmazonRootCA1.pem",
                    cert_reqs=ssl.CERT_REQUIRED,
            ))
        else:
            self.connection = await aio_pika.connect_robust(
                login='ocr_pipeline', password=self.mq_password, host=os.environ.get("MQ_HOST", "54.90.219.231"), port=5672,
                loop=loop,
                timeout=10,
                virtualhost=self.mq_config['virtual_host'],
                heartbeat=self.mq_config['heartbeat']
            )
        self.channel = await self.connection.channel()
        # Declaring queue
        for queue_name in self.supported_queue_names:
            await self.channel.declare_queue(
                queue_name, durable=True
            )
            logging.debug(f'rabbitmq channel declared queue {queue_name}')
        logging.info(f'RabbitMQ Producer Connected')

    async def publish(self, message, queue_name):
        if not self.channel:
            self.channel = await self.connection.channel()
            logging.info('producer recreated a channel')
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue_name
        )

    async def close(self):
        await self.channel.close()
        await self.connection.close()
