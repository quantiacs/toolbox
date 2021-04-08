import gzip, pickle

from qnt.data import get_env
from qnt.log import log_err, log_info


def write(state):
    if state is None:
        return
    path = get_env("OUT_STATE_PATH", "state.out.pickle.gz")
    with gzip.open(path, 'wb') as gz:
        pickle.dump(state, gz)
        log_info("State saved.")


def read(path=None):
    if path is None:
        path = get_env("IN_STATE_PATH", "state.in.pickle.gz")
    try:
        with gzip.open(path, 'rb') as gz:
            res = pickle.load(gz)
            log_info("State loaded.")
            return res
    except Exception as e:
        log_err("Can't load state.", e)
        return None