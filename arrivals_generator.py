import json
import os
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from datetime import datetime, timedelta
from numpy.random import triangular as triang
import numpy as np
import pandas as pd

model_path = '/Users/francescameneghello/Desktop/ConsultaDataMining201618_prf.json'
metadata_file = '/Users/francescameneghello/Desktop/ConsultaDataMining201618_prf_meta.json'


def generate_arrivals(num_instances, start_time):
    with open(model_path, 'r') as fin:
        m = model_from_json(json.load(fin))  # Load model
    with open(metadata_file) as file:
        max_cap = json.load(file)['max_cap']
    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    gen = list()
    n_gen_inst = 0
    while n_gen_inst < num_instances:
        future = pd.date_range(start=start_time, end=(start_time + timedelta(days=30)), freq='H').to_frame(
            name='ds', index=False)
        future['cap'] = max_cap
        future['floor'] = 0
        forecast = m.predict(future)

        def rand_value(x):
            raw_val = triang(x.yhat_lower, x.yhat, x.yhat_upper, size=1)
            raw_val = raw_val[0] if raw_val[0] > 0 else 0
            return raw_val

        forecast['gen'] = forecast.apply(rand_value, axis=1)
        forecast['gen_round'] = np.ceil(forecast['gen'])
        n_gen_inst += np.sum(forecast['gen_round'])
        gen.append(forecast[forecast.gen_round > 0][['ds', 'gen_round']])
        start_time = forecast.ds.max()

    gen = pd.concat(gen, axis=0, ignore_index=True)

    def pp(start, n):
        start_u = int(start.value // 10 ** 9)
        end_u = int((start + timedelta(hours=1)).value // 10 ** 9)
        return pd.to_datetime(np.random.randint(start_u, end_u, int(n)), unit='s').to_frame(name='timestamp')

    gen_cases = list()
    for row in gen.itertuples(index=False):
        gen_cases.append(pp(row.ds, row.gen_round))
    times = pd.concat(gen_cases, axis=0, ignore_index=True)
    times = times.iloc[:num_instances]
    print(times)
    times['caseid'] = times.index + 1
    times['caseid'] = times['caseid'].astype(str)
    times['caseid'] = 'Case' + times['caseid']
    print(times)

num_instances = 10
start_time = '2016-02-01T12:00:00'

generate_arrivals(num_instances, start_time)