import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd

# Load NetCDF file
file = st.file_uploader("Upload NetCDF file", type=["nc"])
if file is not None:
    ds = xr.open_dataset(file)

    # Select variable
    variables = list(ds.data_vars)
    variable = st.selectbox("Select variable", variables)

    # Check lat/lon names
    lat_name = [v for v in ds.dims if 'lat' in v.lower()][0]
    lon_name = [v for v in ds.dims if 'lon' in v.lower()][0]

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Select time
    times = ds['Time'].values if 'Time' in ds else ds['time'].values
    time_index = st.slider("Select time index", 0, len(times)-1, 0)

    # Plot emissions map
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds[variable].isel(Time=time_index).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"{variable} at {times[time_index]}")
    st.pyplot(fig)

    # Show click map (Folium)
    st.subheader("Click on the map to see time series")
    m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=7)
    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]

        # Find nearest grid point
        lat_idx = np.abs(lats - click_lat).argmin()
        lon_idx = np.abs(lons - click_lon).argmin()

        st.write(f"Nearest grid point: lat={lats[lat_idx]:.3f}, lon={lons[lon_idx]:.3f}")

        # Extract time series
        times = pd.to_datetime(times)
        ts = ds[variable][:, lat_idx, lon_idx].values

        # Plot time series
        fig_ts, ax_ts = plt.subplots()
        ax_ts.plot(times, ts)
        ax_ts.set_title(f"Time Series of {variable} at clicked point")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel(variable)
        st.pyplot(fig_ts)

