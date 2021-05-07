import logging
import subprocess

def init_logging(log_file):
    root_logger = logging.getLogger('uc2')
    if not root_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.INFO)  # Hide DEBUG messages from terminal output
        formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stdout_handler)
        root_logger.setLevel(logging.DEBUG)


def get_logger(name):
    return logging.getLogger(f'uc2.{name}')


def log_run(shell_command, logger):
    proc = subprocess.run(shell_command, shell=True, capture_output=True)
    logger.info(f'Called `{proc.args}`')
    if proc.stdout:
        stdout = proc.stdout.decode('utf8').strip()
        for line in stdout.split('\n'):
            logger.info(f'> {line}')
    if proc.stderr:
        stderr = proc.stderr.decode('utf8').strip()
        for line in stderr.split('\n'):
            logger.error(f'> {line}')

