from libs.ops.metrics.metrics_publisher import DurationMetrics, CountMetrics


class Metrics:
    ocr_pipeline_duration = DurationMetrics('ocr-pipeline-duration', 'ocr-pipeline')
    model_duration = DurationMetrics('model-duration', 'ocr-pipeline')
    # download_image_duration = DurationMetrics('download-image-duration', 'network')
    # download_from_s3_duration = DurationMetrics('download-from-s3-duration', 'network')
    download_image_fail = CountMetrics('download-image-fail', 'network')
    # upload_to_s3_duration = DurationMetrics('upload-to-s3-duration', 'network')
    unexpected_error = CountMetrics('ocr-pipeline-unexpected-error', 'ocr-pipeline')
    initialization_error = CountMetrics('initialization-error', 'ocr-pipeline')

    receive_count = CountMetrics('receive-count', 'ocr-pipeline')
    fail_count = CountMetrics('fail-count', 'ocr-pipeline')

    metrics = [
        ocr_pipeline_duration, model_duration, unexpected_error,
        initialization_error, download_image_fail, fail_count, receive_count
    ]
