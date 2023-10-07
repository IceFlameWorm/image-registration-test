import asyncio
import botocore.exceptions
import json
import logging
import os
import sys
import traceback

from aiobotocore.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

sqs_logger = logging.getLogger(__name__)
sqs_logger.setLevel(logging.INFO)


class AsyncConsumer:
    def __init__(self, superworker, publish_metrics_task=None):
        self.env = os.environ['ENV']
        self.session = get_session()
        self.process_message = superworker.process_message
        # for OCR API
        self.input_queue_name = superworker.input_topic + f'-{self.env}'
        self.output_queue_name = superworker.output_topic + f'-{self.env}'
        # for MQ Mode
        # priority queue
        self.input_430_priority_queue_list = []
        self.input_430_student_id_priority_queue_list= []
        priority_queue_430_count = 1
        priority_queue_430_student_id_count = 1
        for order in range(priority_queue_430_count):
            self.input_430_priority_queue_list.append(f'{superworker.input_topic}-430-{order}-{self.env}')
        for order in range(priority_queue_430_student_id_count):
            self.input_430_student_id_priority_queue_list.append(f'{superworker.input_topic}-430-student-id-{order}-{self.env}')
        
        # main queue
        self.input_430_priority_queue_list.append(f'{superworker.input_topic}-430-{self.env}')
        self.input_430_student_id_priority_queue_list.append(f'{superworker.input_topic}-430-student-id-{self.env}')


        self.input_queue_name_430_init = superworker.input_topic + '-430-init' + f'-{self.env}'
        self.output_queue_name_430 = superworker.output_topic + '-430' + f'-{self.env}'
        self.output_queue_name_430_student_id = superworker.output_topic + '-430-student-id' + f'-{self.env}'
        self.output_queue_name_430_init = superworker.output_topic + '-430-init' + f'-{self.env}'
        if self.env == 'preprod-dev':
            self.input_queue_name_qa = superworker.input_topic + '-430-qa'
            self.output_queue_name_qa = superworker.output_topic + '-430-qa'
            self.input_queue_name_student_id_qa = superworker.input_topic + '-430-student-id-qa'
            self.output_queue_name_student_id_qa = superworker.output_topic + '-430-student-id-qa'
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(self.handle_exception)
        try:
            loop.run_until_complete(self.start_all(loop, publish_metrics_task))
            loop.run_forever()
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            sqs_logger.error(str(e) + '\n{}'.format(traceback.format_exception(exc_type, exc_value, exc_tb)))
            loop.stop()
            sqs_logger.info("Stopped to process messages")

    async def start_all(self, loop, publish_metrics_task):
        if publish_metrics_task:
            task_func = publish_metrics_task.pop(0)
            task_args = publish_metrics_task
            loop.create_task(task_func(*task_args))
        await asyncio.gather(self.main())

    def handle_exception(self, loop, context):
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get("exception", context["message"])
        sqs_logger.error(f"Caught exception: {msg}")

    async def consume(self, input_queue_name, output_queue_name):
        while True:
            try:
                async with self.session.create_client('sqs') as client:
                    try:
                        response = await client.get_queue_url(QueueName=input_queue_name)
                    except botocore.exceptions.ClientError as err:
                        if err.response['Error']['Code'] == \
                                'AWS.SimpleQueueService.NonExistentQueue':
                            sqs_logger.error(f"Queue {input_queue_name} does not exist")
                            sys.exit(1)
                        else:
                            raise

                    queue_url = response['QueueUrl']

                    sqs_logger.info(f'start consuming {input_queue_name}...')
                    while True:
                        # This loop wont spin really fast as there is
                        # essentially a sleep in the receive_message call
                        response = await client.receive_message(
                            QueueUrl=queue_url
                        )

                        if 'Messages' in response:
                            for msg in response['Messages']:
                                await self.process_message(msg, output_queue_name)
                                # Need to remove msg from queue or else it'll reappear
                                await client.delete_message(
                                    QueueUrl=queue_url,
                                    ReceiptHandle=msg['ReceiptHandle']
                                )
                        else:
                            sqs_logger.debug('No messages in queue')
            except botocore.exceptions.ClientError as e:
                sqs_logger.error(e)
                sqs_logger.info('refresh consumer session client')

    async def consume_priority_queues(self, priority_queue_list, output_queue_name):
        while True:
            try:
                async with self.session.create_client('sqs') as client:
                    priority_queue_url_list = []
                    for queue in priority_queue_list:
                        try:
                            response = await client.get_queue_url(QueueName=queue)
                        except botocore.exceptions.ClientError as err:
                            if err.response['Error']['Code'] == \
                                    'AWS.SimpleQueueService.NonExistentQueue':
                                sqs_logger.error(f"Queue {queue} does not exist")
                                sys.exit(1)
                            else:
                                raise
                        priority_queue_url_list.append(response['QueueUrl'])

                    while True:
                        for idx, queue_url in enumerate(priority_queue_url_list):
                            sqs_logger.debug(f'consuming {priority_queue_list[idx]}...')
                        
                            # This loop wont spin really fast as there is
                            # essentially a sleep in the receive_message call
                            response = await client.receive_message(
                                QueueUrl=queue_url
                            )

                            if 'Messages' in response:
                                for msg in response['Messages']:
                                    await self.process_message(msg, output_queue_name, idx)
                                    # Need to remove msg from queue or else it'll reappear
                                    await client.delete_message(
                                        QueueUrl=queue_url,
                                        ReceiptHandle=msg['ReceiptHandle']
                                    )
                                break
                            else:
                                sqs_logger.debug(f'No messages in queue {priority_queue_list[idx]}')
            except botocore.exceptions.ClientError as e:
                sqs_logger.error(e)
                sqs_logger.info('refresh consumer session client')

    
    async def main(self):
        if os.environ.get('APP', '') == '430':
            if os.environ.get('PRIORITY_QUEUE', 'False') == 'True':
                await self.consume_priority_queues(self.input_430_priority_queue_list, self.output_queue_name_430)
            else:
                await self.consume(self.input_430_priority_queue_list[-1], self.output_queue_name_430)
        elif self.env == 'preprod-dev' and os.environ.get('APP', '') == '430-qa':
            await self.consume(self.input_queue_name_qa, self.output_queue_name_qa)
        elif self.env == 'preprod-dev' and os.environ.get('APP', '') == '430-student-id-qa':
            await self.consume(self.input_queue_name_student_id_qa, self.output_queue_name_student_id_qa)
        elif os.environ.get('APP', '') == '430-init':
            await self.consume(self.input_queue_name_430_init, self.output_queue_name_430_init)
        elif os.environ.get('APP', '') == '430-student-id':
            if os.environ.get('PRIORITY_QUEUE', 'False') == 'True':
                await self.consume_priority_queues(self.input_430_student_id_priority_queue_list, self.output_queue_name_430_student_id)
            else:
                await self.consume(self.input_430_student_id_priority_queue_list[-1], self.output_queue_name_430_student_id)
        else:
            await self.consume(self.input_queue_name, self.output_queue_name)


class AsyncProducer:
    def __init__(self):
        self.session = get_session()

    async def publish(self, message, queue_name):
        async with self.session.create_client('sqs') as client:
            try:
                response = await client.get_queue_url(QueueName=queue_name)
            except botocore.exceptions.ClientError as err:
                if err.response['Error']['Code'] == \
                        'AWS.SimpleQueueService.NonExistentQueue':
                    sqs_logger.error(f"Queue {queue_name} does not exist")
                    sys.exit(1)
                else:
                    raise

            queue_url = response['QueueUrl']    
            await client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message, separators=(',', ':'))
            )
            sqs_logger.info(f'sent "{message}" to queue {queue_name}')
