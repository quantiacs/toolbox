import qnt.stats as qnstats
import xarray as xr
import numpy as np


def drop_bad_days(weights, max_weight = 0.049):
    exposure = qnstats.calc_exposure(weights)
    return weights.where(exposure.max('asset') < max_weight).fillna(0)


def mix_weights(primary, secondary, max_weight = 0.049):
    primary, secondary = xr.align(primary, secondary, join='outer')

    primary = primary.fillna(0)
    secondary = secondary.fillna(0)

    primary_exposure = qnstats.calc_exposure(primary)
    primary_max_exposure = primary_exposure.max('asset')
    primary_abs_sum = abs(primary).sum('asset')

    secondary_exposure = qnstats.calc_exposure(secondary)
    secondary_max_exposure = secondary_exposure.max('asset')
    secondary_abs_sum = abs(secondary).sum('asset')

    # formula
    k = primary_abs_sum * (primary_max_exposure - max_weight) / \
        (secondary_abs_sum * ( max_weight - secondary_max_exposure) )

    k = k.where(k > 0, 0) # k > 0

    mix = primary + secondary * k
    # normalization
    sum = abs(mix).sum('asset')
    sum = sum.where(sum > 1, 1)
    mix = mix/sum

    return mix


def cut_big_positions(weights, max_weight = 0.049):
    return weights.where(abs(weights) > max_weight, np.sign(weights)*max_weight, weights)
