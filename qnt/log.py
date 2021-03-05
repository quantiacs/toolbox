import sys


orig_print = print
info_enabled = True
err_enabled = True


def log_info(*args):
    if info_enabled:
        orig_print(*args, flush=True)


def log_err(*args):
    if err_enabled:
        orig_print(*args, file=sys.stderr, flush=True)


class Settings(object):
    def __init__(self, info=True, err=True):
        self.info = info
        self.err = err

    def __enter__(self):
        global info_enabled, err_enabled
        self.old_info = info_enabled
        self.old_err = err_enabled
        info_enabled = self.info
        err_enabled = self.err

    def __exit__(self, exc_type, exc_val, exc_tb):
        global info_enabled, err_enabled
        info_enabled = self.old_info
        err_enabled = self.old_err
