import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

from pyproj import Proj, Transformer

def generate_latlon_grid_from_griddesc():
    # Grid info from GRIDDESC
    x0 = -505568.188  # origin x in meters
    y0 = 1994348.750  # origin y in meters
    dx = 27000.0      # cell size in x
    dy = 27000.0      # cell size in y
    nx = 29
    ny = 29

    # Center of grid cells
    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(x, y)

    # Define Mercator projection centered at 106.678
    mercator_proj = Proj(proj='merc', lat_ts=0.0, lon_0=106.678, ellps='WGS84')

    # Convert to lat/lon
    transformer = Transformer.from_proj(mercator_proj, 'epsg:4326', always_xy=True)
    lon2d, lat2d = transformer.transform(X, Y)

    return lon2d, lat2d

st.set_page_config(layout="wide")
st.title("CMAQ Concentration Viewer")

# --- Input ---
nc_path = st.sidebar.text_input("Path to CMAQ NetCDF file:", 
    "/mnt/scratch_lustre/duch/CMAQ_installation/data/WRF/hanoi/2022-07-01/d01/CCTM_CONC_v54_intel_d01_20220701.nc")

if nc_path:
    try:
        ds = xr.open_dataset(nc_path)
        st.sidebar.success("CMAQ file loaded.")

        # List pollutant variables (exclude TFLAG)
        variables = [v for v in ds.data_vars if v != "TFLAG"]
        variable = st.sidebar.selectbox("Select variable", variables)

        # Convert TFLAG to datetime
        tflag = ds["TFLAG"].values[:, 0, :]  # shape: (TSTEP, 2)
        times = []
        for t in tflag:
            yyyyddd, hhmmss = t
            year = int(str(yyyyddd)[:4])
            doy = int(str(yyyyddd)[4:])
            hour = int(str(hhmmss).zfill(6)[:2])
            minute = int(str(hhmmss).zfill(6)[2:4])
            second = int(str(hhmmss).zfill(6)[4:])
            dt = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
            times.append(dt)
        
        # Time selection
        time_index = st.sidebar.slider("Time Index", 0, len(times) - 1, 0)
        st.sidebar.write(f"Selected time: {times[time_index]}")

        # Plot surface layer
        st.subheader(f"{variable} at {times[time_index]} (Layer 1)")
        data = ds[variable].isel(TSTEP=time_index, LAY=0)  # Surface layer

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        lon2d, lat2d = generate_latlon_grid_from_griddesc()
        mesh = ax.pcolormesh(lon2d, lat2d, data.values, transform=ccrs.PlateCarree(), cmap='viridis')
        #mesh = ax.pcolormesh(data["COL"], data["ROW"], data.values, transform=ccrs.PlateCarree(), cmap='viridis')
        plt.colorbar(mesh, ax=ax, label=ds[variable].units)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)

        # Add lat/lon gridlines with labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f"{variable} at {times[time_index]}")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load or plot file: {e}")

