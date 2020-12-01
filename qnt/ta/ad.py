from qnt.ta.ndadapter import NdType, nd_to_1d_universal_adapter
from qnt.ta.change import change
import numpy as np


def ad_ratio_np(prices: np.ndarray) -> np.ndarray:
    """
     Advance/Decline Ratio (numpy)
    :param prices: last dimension - time
    :return:
    """
    ch = change(prices, 1)
    ch = np.nan_to_num(ch)
    advance = np.where(ch > 0, 1, 0).sum(axis=0)
    decline = np.where(ch < 0, 1, 0).sum(axis=0)
    return advance / decline


def ad_ratio(prices: NdType) -> NdType:
    """
    Advance/Decline Ratio
    :param prices:
    :return:
    """
    return nd_to_1d_universal_adapter(ad_ratio_np, (prices,), ())


def ad_line_np(prices: np.ndarray) -> np.ndarray:
    """
    Advance/Decline Line (numpy)
    :param prices: last dimension - time
    :return:
    """
    ch = change(prices, 1)
    ch = np.nan_to_num(ch)
    advance = np.where(ch > 0, 1, 0).sum(axis=0)
    decline = np.where(ch < 0, 1, 0).sum(axis=0)
    return (advance - decline).cumsum()


def ad_line(prices: NdType) -> NdType:
    """
    Advance/Decline Ratio
    :param prices:
    :return:
    """
    if isinstance(prices, np.ndarray):
        return ad_line_np(prices)
    return nd_to_1d_universal_adapter(ad_line_np, (prices,), ())


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field="close")

    adr = ad_ratio(data)
    adr_pd = ad_ratio(data.to_pandas())
    print(adr.to_pandas() - adr_pd.T)

    adl = ad_line(data)

    import matplotlib.pyplot as plt

    plt.plot(adr.coords[ds.TIME].values, adr.values, 'r')
    plt.show()

    plt.plot(adl.coords[ds.TIME].values, adl.values, 'g')
    plt.show()
