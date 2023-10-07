import pika
import functools
import threading
from pipeline_config import PipelineConfig
from credentials import MQ_PASSWORD


class AMQPConsumer:
    def __init__(self, n_times=-1):
        self.config = PipelineConfig()
        credentials = pika.PlainCredentials('ocr_pipeline', MQ_PASSWORD)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=3600,
                                                                            blocked_connection_timeout=300,
                                                                            host=self.config.QUEUE_IP,
                                                                            port=self.config.QUEUE_PORT,
                                                                            virtual_host='/',
                                                                            credentials=credentials))
        self.channel = self.connection.channel()
        self.counter = 0
        self.n_times = n_times

    def queue_declare(self, queue_name):
        """
        Declare queue, create if doesn't exist
        :param queue_name: str
        :return:https://stackoverflow.com/questions/1187970/how-to-exit-from-python-without-traceback
        """
        self.channel.queue_declare(queue=queue_name, durable=True)

    def on_message(self, channel, method_frame, header_frame, body, args):
        callback = args
        delivery_tag = method_frame.delivery_tag
        t = threading.Thread(target=callback, args=(self.connection, channel, delivery_tag, body))
        t.start()

    def consume(self, queue_name, callback):
        """
        consume message from specific queue
        :param queue_name: str
        :param callback: The function to call when consuming with the signature (channel, method, properties, body)
        :return:
        """
        on_message_callback = functools.partial(self.on_message, args=callback)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name,
                                   on_message_callback=on_message_callback)
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
