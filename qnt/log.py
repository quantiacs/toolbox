import sys


orig_print = print


def log_info(*args):
    orig_print(*args, flush=True)


def log_err(*args):
    orig_print(*args, file=sys.stderr, flush=True)