import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import requests
from sklearn.metrics import mean_squared_error

sites = {
    "Richmond": {"lat": -33.6, "lon": 150.75},
    "Liverpool": {"lat": -33.92, "lon": 150.92},
    "Randwick": {"lat": -33.92, "lon": 151.25},
    # Add more if needed
}

def extract_model_timeseries(ds, varname, sites):
    model_timeseries = {}

    # Convert site lat/lon to 1D arrays
    lat = ds['lat'].values
    lon = ds['lon'].values
    time = ds['time'].values

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    tree = cKDTree(points)

    for name, loc in sites.items():
        dist, idx = tree.query([loc["lat"], loc["lon"]])
        iy, ix = np.unravel_index(idx, lat_grid.shape)
        ts = ds[varname][:, iy, ix].values
        model_timeseries[name] = pd.Series(ts, index=pd.to_datetime(time))

    return model_timeseries


def get_obs_timeseries(site_name, start_date, end_date, parameter):
    url = f"https://example-observation-api.org/data?site={site_name}&parameter={parameter}&start={start_date}&end={end_date}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        timestamps = [pd.to_datetime(d['timestamp']) for d in data]
        values = [d['value'] for d in data]
        return pd.Series(values, index=timestamps)
    else:
        print(f"Error fetching data for {site_name}")
        return pd.Series()


def compute_metrics(model_series, obs_series):
    df = pd.concat([model_series, obs_series], axis=1).dropna()
    model = df.iloc[:, 0]
    obs = df.iloc[:, 1]

    correlation = model.corr(obs)
    rmse = mean_squared_error(obs, model, squared=False)
    mb = (model - obs).mean()
    me = np.abs(model - obs).mean()

    return {
        "Correlation": correlation,
        "RMSE": rmse,
        "Mean Bias (MB)": mb,
        "Mean Error (ME)": me,
    }

import matplotlib.pyplot as plt

def plot_comparison(model_series, obs_series, site_name):
    plt.figure(figsize=(12,4))
    plt.plot(model_series, label='Model')
    plt.plot(obs_series, label='Observed', linestyle='--')
    plt.title(f"Time Series at {site_name}")
    plt.xlabel("Time")
    plt.ylabel("Concentration (µg/m³ or ppb)")
    plt.legend()
    plt.tight_layout()
    plt.show()

ds = xr.open_dataset("your_cmaq_output.nc")
varname = "no2"  # or "o3", "pm25", etc.
start_date = "2025-08-01"
end_date = "2025-08-02"

model_data = extract_model_timeseries(ds, varname, sites)

for site in sites:
    obs_series = get_obs_timeseries(site, start_date, end_date, varname)
    model_series = model_data[site]

    metrics = compute_metrics(model_series, obs_series)
    print(f"Metrics at {site}: {metrics}")

    plot_comparison(model_series, obs_series, site)

