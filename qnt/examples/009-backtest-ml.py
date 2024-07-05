"""
This example shows how to use the backtester with machine learning.
Most of ML libraries supports batching and allows to calculate multiple values in one pass.
So the backtester was adapted to this feature in order to improve the execution speed.
"""
import logging

import pandas as pd
import xarray as xr
import numpy as np

import qnt.backtester as qnbt
import qnt.ta as qnta


# constructor for ML model, you can use almost any ML classifier
def create_model():
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import SGDClassifier, RidgeClassifier
    import random
    # We will use model combined from RidgeClassifier and SGDClassifier.
    # Also we use several random seeds to reduce impact of random.
    classifiers = []
    r = random.Random(13)
    for i in range(42):
        classifiers.append(('ridge' + str(i), RidgeClassifier(random_state=r.randint(0, pow(2, 32) - 1)),))
        classifiers.append(('sgd' + str(i), SGDClassifier(random_state=r.randint(0, pow(2, 32) - 1)),))
    model = VotingClassifier(classifiers)

    return model


# Builds features for learning.
def get_features(data):
    trend = qnta.roc(qnta.lwma(data.sel(field="close"), 90), 1)

    k, d = qnta.stochastic(data.sel(field="high"), data.sel(field="low"), data.sel(field="close"), 21)

    volatility = qnta.tr(data.sel(field="high"), data.sel(field="low"), data.sel(field="close"))
    volatility = volatility / data.sel(field="close")
    volatility = qnta.lwma(volatility, 21)

    volume = data.sel(field="vol")
    volume = qnta.sma(volume, 7) / qnta.sma(volume, 80)
    volume = volume.where(np.isfinite(volume), 0)

    # combine features to one array
    result = xr.concat(
        [trend, d, volatility, volume ],
        pd.Index(
            ['trend', 'stochastic_d', 'volatility', 'volume'],
            name='field'
        )
    )
    return result.transpose('time', 'field', 'asset')


def get_target_classes(data):
    # for classifiers, you need to set classes
    # if 1 then the price will rise tomorrow

    price_current = data.sel(field="close")
    price_future = qnta.shift(price_current, -1)

    class_positive = 1
    class_negative = 0

    target_is_price_up = xr.where(price_future > price_current, class_positive, class_negative)
    return target_is_price_up


# creates and trains models
# one ML model per asset
def create_and_train_models(data):
    asset_name_all = data.coords['asset'].values

    data = data.sel(time=slice('2013-05-01',None))  # cut the head before 2013-05-01 (a lot of noise)

    features_all = get_features(data)
    target_all = get_target_classes(data)

    models = dict()

    for asset_name in asset_name_all:
        target_cur = target_all.sel(asset=asset_name).dropna('time', how='any')
        features_cur = features_all.sel(asset=asset_name).dropna('time', how='any')

        # align features and targets
        target_for_learn_df, feature_for_learn_df = xr.align(target_cur, features_cur, join='inner')

        if len(features_cur.time) < 10:
            # not enough points for training
            continue

        model = create_model()
        try:
            model.fit(feature_for_learn_df.values, target_for_learn_df)
            models[asset_name] = model
        except KeyboardInterrupt as e:
            raise e
        except:
            logging.exception("model training failed")

    return models


# performs prediction and generates output weights
# it generates output for several days in order to speed up the evaluation
def predict(models, data):
    asset_name_all = data.coords['asset'].values
    weights = xr.zeros_like(data.sel(field="close"))
    for asset_name in asset_name_all:
        if asset_name in models:
            model = models[asset_name]
            features_all = get_features(data)
            features_cur = features_all.sel(asset=asset_name).dropna('time',how='any')
            if len(features_cur.time) < 1:
                continue
            try:
                weights.loc[dict(asset=asset_name,time=features_cur.time.values)] = model.predict(features_cur.values)
            except KeyboardInterrupt as e:
                raise e
            except:
                logging.exception("model failed")
    return weights


# runs the backtester
weights = qnbt.backtest_ml(
    train=create_and_train_models,
    predict=predict,
    train_period=4*365,   # the data length for training in calendar days
    retrain_interval=365,  # how often we have to retrain models (calendar days)
    retrain_interval_after_submit=1, # how often retrain models after submit during evaluation (calendar days)
    predict_each_day=False,  # is it necessary to call predict for each day.
                             # Set true if you suspect that get_features is looking forward.
    competition_type='cryptofutures',  # competition type
    lookback_period=365,  # how many calendars the predict function needs to generate the output
    start_date='2014-01-01',  # backtest start date
    build_plots=True  # do you need the chart?
)
