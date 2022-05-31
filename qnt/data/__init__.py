from .stocks import load_list as stocks_load_list
from .stocks import load_data as stocks_load_data
from .stocks import load_ndx_list as stocks_load_ndx_list
from .stocks import load_ndx_data as stocks_load_ndx_data
from .stocks import load_origin_data as stocks_load_origin_data
from .stocks import restore_origin_data as stocks_restore_origin_data
from .stocks import adjust_by_splits as stocks_adjust_by_splits

from .blsgov import load_db_list as blsgov_load_db_list
from .blsgov import load_db_meta as blsgov_load_db_meta
from .blsgov import load_series_list as blsgov_load_series_list
from .blsgov import load_series_aspect as blsgov_load_series_aspect
from .blsgov import load_series_data as blsgov_load_series_data

from .secgov import load_forms as secgov_load_forms
from .secgov import load_facts as secgov_load_facts
from .secgov_indicators import load_indicators as secgov_load_indicators
from .secgov_fundamental import load_indicators_for as secgov_load_indicators_for
from .secgov_fundamental import get_all_indicator_names as secgov_get_all_indicator_names
from .secgov_fundamental import get_complex_indicator_names as secgov_get_complex_indicator_names

from .crypto import load_data as crypto_load_data
from .cryptofutures import load_data as cryptofutures_load_data

from .futures import load_list as futures_load_list
from .futures import load_data as futures_load_data

from .index import major_load_list as index_major_load_list
from .index import major_load_data as index_major_load_data
from .index import load_list as index_load_list
from .index import load_data as index_load_data

from .imf import load_currency_list as imf_load_currency_list
from .imf import load_currency_data as imf_load_currency_data
from .imf import load_commodity_list as imf_load_commodity_list
from .imf import load_commodity_data as imf_load_commodity_data

from .blockchaincom import load_list as blockchaincom_load_list
from .blockchaincom import load_data as blockchaincom_load_data

from .cryptodaily import load_data as cryptodaily_load_data

from .common import Fields, f, Dimensions, ds, get_env, deprecated_wrap

from ..output import write as write_output
from ..output import normalize as sort_and_crop_output
from ..output import clean as clean_output
from ..output import check as check_output


def load_data_by_type(data_type, **kwargs):
    if data_type == 'stocks' or data_type == 'stocks_long':
        return stocks_load_data(**kwargs)
    if data_type == 'stocks_nasdaq100':
        return stocks_load_ndx_data(**kwargs)
    elif data_type == 'futures':
        return futures_load_data(**kwargs)
    elif data_type == 'crypto':
        return crypto_load_data(**kwargs)
    elif data_type == 'crypto_futures' or data_type == 'cryptofutures':
        return cryptofutures_load_data(**kwargs)
    elif data_type == 'crypto_daily' or data_type == 'cryptodaily'\
            or data_type == 'crypto_daily_long' or data_type == 'crypto_daily_long_short':
        return cryptodaily_load_data(**kwargs)
    else:
        raise Exception("Wrong data_type.")


load_assets = deprecated_wrap(stocks_load_list)
load_data = deprecated_wrap(stocks_load_data)
load_origin_data = deprecated_wrap(stocks_load_origin_data)
restore_origin_data = deprecated_wrap(stocks_restore_origin_data)
adjust_by_splits = deprecated_wrap(stocks_adjust_by_splits)

load_blsgov_db_list = deprecated_wrap(blsgov_load_db_list)
load_blsgov_db_meta = deprecated_wrap(blsgov_load_db_meta)
load_blsgov_series_list = deprecated_wrap(blsgov_load_series_list)
load_blsgov_series_aspect = deprecated_wrap(blsgov_load_series_aspect)
load_blsgov_series_data = deprecated_wrap(blsgov_load_series_data)

load_secgov_forms = deprecated_wrap(secgov_load_forms)
load_secgov_facts = deprecated_wrap(secgov_load_facts)

load_cryptocurrency_data = deprecated_wrap(crypto_load_data)

load_futures_list = deprecated_wrap(futures_load_list)
load_futures_data = deprecated_wrap(futures_load_data)

load_index_data = deprecated_wrap(index_load_data)
load_index_list = deprecated_wrap(index_load_list)
load_major_index_list = deprecated_wrap(index_major_load_list)
load_major_index_data = deprecated_wrap(index_major_load_data)

write_output = deprecated_wrap(write_output)
sort_and_crop_output = deprecated_wrap(sort_and_crop_output)
check_output = deprecated_wrap(check_output)
clean_output = deprecated_wrap(clean_output)
