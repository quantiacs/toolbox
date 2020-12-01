import qnt.data.common as qndc
import qnt.data.id_translation as idt
idt.USE_ID_TRANSLATION = False
qndc.BASE_URL = 'http://127.0.0.1:7070/'

import qnt.data as qndata

import time
import datetime as dt

dbs = qndata.load_blsgov_db_list()
print(dbs)

db_meta = qndata.load_blsgov_db_meta('CX')
print(db_meta)

for s in qndata.load_blsgov_series_list('CX'):
    print(s)

ds = qndata.load_blsgov_series_data('CXUWOMENSLB1407M', min_date='1900-01-01')

print(ds)


ds = qndata.load_blsgov_series_aspect('CXUWOMENSLB1407M')

print(ds)


