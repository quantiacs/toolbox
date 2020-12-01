import qnt.data as qndata
import time
import datetime as dt

i = 0
j = 0
st = time.time()
# for f in qndata.load_secgov_forms(
#         types=['10-Q'],
#         # facts=[
#         #     'us-gaap:EarningsPerShareDiluted',
#         #     'us-gaap:Liabilities',
#         #     'us-gaap:Assets',
#         #     'us-gaap:CommonStockSharesOutstanding'
#         # ],
#         # skip_segment=True,
#         tail=dt.timedelta(days=365)
# ):
#     # print(f['url'], len(f['facts']))
#     print(i, j, f['date'], time.time() - st)
#     i += 1
#     j += len(f['facts'])
#qndata.BASE_URL = 'http://127.0.0.1:8001/'

for f in qndata.load_secgov_facts(
        ciks=['1800'],
        #types=['10-Q'],
        facts=[
             'us-gaap:EarningsPerShareDiluted',
             'us-gaap:Liabilities',
             'us-gaap:Assets',
             'us-gaap:CommonStockSharesOutstanding'
        ],
        skip_segment=True,
        period='Q',
        columns=['fact_name', 'form_date', 'value', 'period_length'],
        tail=dt.timedelta(days=365)
):
    print(f)
    i += 1