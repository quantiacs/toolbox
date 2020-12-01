import qnt.data as qndata  # data loading and manipulation
import qnt.forward_looking as qnfl  # forward looking checking

qndata.MAX_DATE_LIMIT = None
import datetime

data1 = qndata.load_data(
    min_date="2015-01-01",
    # max_date="2019-03-04", # You should not limit max_date for final calculations!
    dims=("time", "field", "asset"),
    forward_order=True
)

last_date = datetime.datetime.now().date()
last_date = last_date - datetime.timedelta(days=182)
qndata.MAX_DATE_LIMIT = last_date

data2 = qndata.load_data(
    min_date="2015-01-01",
    # max_date="2019-03-04", # You should not limit max_date for final calculations!
    dims=("time", "field", "asset"),
    forward_order=True
)

qndata.MAX_DATE_LIMIT = None

print(data1)
print(data2)

qnfl.check_forward_looking(
    data2.sel(field='close').where(data2.sel(field='is_liquid') > 0),
    data1.sel(field='close').where(data1.sel(field='is_liquid') > 0)
)
