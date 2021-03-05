import qnt.data as qndata
import qnt.ta as qnta
import qnt.optimizer as qno
import xarray as xr


data = qndata.futures.load_data(min_date="2005-01-01",  forward_order=True, dims=("time", "field", "asset"))


def strategy(data, wma_period=290, roc_period=35):
    wma = qnta.lwma(data.sel(field='close'), wma_period)
    sroc = qnta.roc(wma, roc_period)
    weights = xr.where(sroc > 0, 1, 0)
    weights = weights / len(data.asset)
    return weights


result = qno.optimize_strategy(
    data,
    strategy,
    qno.full_range_args_generator(
        wma_period=range(10, 150, 5),
        roc_period=range(5, 100, 5)
    ),
    workers=8
)
qno.build_plot(result)
print(result['best_iteration'])