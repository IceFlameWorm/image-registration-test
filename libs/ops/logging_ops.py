import logging


def gen_logging_string(module_name, img_id, log_name, message):
    log_format_string = '[{}] [{}] [{}] [{}]'.format(module_name, img_id, log_name, message)
    return log_format_string