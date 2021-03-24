import sys


orig_print = print
info_enabled = True
err_enabled = True
err2info = False


def log_info(*args):
    if info_enabled:
        orig_print(*args, flush=True)


def log_err(*args):
    if err_enabled:
        orig_print(*args, file=sys.stderr if not err2info else sys.stdout, flush=True)


class Settings(object):
    def __init__(self, info=None, err=None, err2info=None):
        self.info = info
        self.err = err
        self.err2info = err2info

    def __enter__(self):
        global info_enabled, err_enabled, err2info
        self.old_info = info_enabled
        self.old_err = err_enabled
        self.old_err2info = err2info
        if self.info is not None:
            info_enabled = self.info
        if self.err is not None:
            err_enabled = self.err
        if self.err2info is not None:
            err2info = self.err2info

    def __exit__(self, exc_type, exc_val, exc_tb):
        global info_enabled, err_enabled, err2info
        if self.info is not None:
            info_enabled = self.old_info
        if self.err is not None:
            err_enabled = self.old_err
        if self.err2info is not None:
            err2info = self.old_err2info
