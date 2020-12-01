import qnt.data.common as qtc
import qnt.data as qndata
import time
import datetime as dt

qtc.BASE_URL = 'http://127.0.0.1:8001/'

idx_list = qndata.load_index_list( tail=dt.timedelta(days=365))

print(idx_list)

idx_data = qndata.load_index_data(assets=["RUT"], forward_order=True, tail=dt.timedelta(days=365))

print(idx_data)

major_idx_list = qndata.load_major_index_list()

print(major_idx_list)

major_idx_data = qndata.load_major_index_data(tail=dt.timedelta(days=365), forward_order=True)

print(major_idx_data)