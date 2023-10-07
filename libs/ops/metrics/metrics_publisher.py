import asyncio
import sys
import logging
import os
import boto3

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class DurationMetrics:
    def __init__(self, metric_name, component_name):
        self.duration_min = sys.maxsize
        self.duration_max = 0
        self.duration_counts = 0
        self.duration_sum = 0
        self.metric_name = metric_name
        self.component_name = component_name

    def add_duration(self, duration):
        self.duration_counts += 1
        self.duration_max = max(self.duration_max, duration)
        self.duration_min = min(self.duration_min, duration)
        self.duration_sum += duration

    def reset(self):
        self.duration_counts = 0
        self.duration_max = 0
        self.duration_min = sys.maxsize
        self.duration_sum = 0


class CountMetrics:
    def __init__(self, metric_name, component_name):
        self.count = 0
        self.metric_name = metric_name
        self.component_name = component_name

    def add_count(self):
        self.count += 1

    def reset(self):
        self.count = 0


class MetricsPublisher:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')

    async def publish_periodic(self, metrics_instances):
        """publish_periodic
        Publish metric every interval
        """
        while True:
            await asyncio.sleep(60)  # publish interval
            self.publish_metrics(metrics_instances)

    def _prepare_metrics(self, metrics_instances, metric_data):
        for metrics_instance in metrics_instances:
            if isinstance(metrics_instance, DurationMetrics) and metrics_instance.duration_counts > 0:
                if metrics_instance.duration_counts > 1:
                    logging.warning(f'metric {metrics_instance.metric_name} has multiple data points at one time')
                metric_data.append({
                    'MetricName': metrics_instance.metric_name,
                    'Dimensions': [{
                        'Name': 'component',
                        'Value': metrics_instance.component_name
                    }],
                    'StatisticValues': {
                        'Minimum': metrics_instance.duration_min,
                        'Maximum': metrics_instance.duration_max,
                        'SampleCount': metrics_instance.duration_counts,
                        'Sum': metrics_instance.duration_sum
                    },
                    'Unit': 'Seconds'
                })

            elif isinstance(metrics_instance, CountMetrics):
                if metrics_instance.count > 1:
                    logging.warning(f'metric {metrics_instance.metric_name} has multiple data points at one time')
                metric_data.append({
                    'MetricName': metrics_instance.metric_name,
                    'Dimensions': [{
                        'Name': 'component',
                        'Value': metrics_instance.component_name
                    }],
                    'Value': metrics_instance.count,
                    'Unit': 'Count'
                })

            metrics_instance.reset()

    def publish_metrics(self, metrics_instances, priority_index=None):
        metric_data = []
        self._prepare_metrics(metrics_instances, metric_data)
        if priority_index:
            namespace = f'LAI/{os.environ["ENV"].upper()}/OCR_PIPELINE/{os.environ.get("APP", "").upper()}/P{priority_index}'
        else:
            namespace = f'LAI/{os.environ["ENV"].upper()}/OCR_PIPELINE/{os.environ.get("APP", "").upper()}'
        if metric_data:
            try:
                self.cloudwatch.put_metric_data(
                    Namespace=namespace, MetricData=metric_data
                )
                # logging.info("Published metrics : {}".format(len(metric_data)))
            except Exception as e:
                logging.warning("Failed to publish metrics {}: {}".format(metric_data, str(e)))
