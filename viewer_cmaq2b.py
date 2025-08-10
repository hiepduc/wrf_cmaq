import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
import folium
from streamlit_folium import st_folium

# --- Function to generate lat/lon grid from GRIDDESC ---
def generate_latlon_grid_from_griddesc():
    x0 = -505568.188
    y0 = 1994348.750
    dx = 27000.0
    dy = 27000.0
    nx = 29
    ny = 29

    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(x, y)

    mercator_proj = Proj(proj='merc', lat_ts=0.0, lon_0=106.678, ellps='WGS84')
    transformer = Transformer.from_proj(mercator_proj, 'epsg:4326', always_xy=True)
    lon2d, lat2d = transformer.transform(X, Y)

    return lon2d, lat2d

# --- Streamlit app ---
st.set_page_config(layout="wide")
st.title("CMAQ Concentration Viewer with Time Series")

nc_path = st.sidebar.text_input("Path to CMAQ NetCDF file:",
    "/mnt/scratch_lustre/duch/CMAQ_installation/data/WRF/hanoi/2022-07-01/d01/CCTM_CONC_v54_intel_d01_20220701.nc")

if nc_path:
    try:
        ds = xr.open_dataset(nc_path)
        st.sidebar.success("CMAQ file loaded.")

        variables = [v for v in ds.data_vars if v != "TFLAG"]
        variable = st.sidebar.selectbox("Select variable", variables)

        tflag = ds["TFLAG"].values[:, 0, :]
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

        time_index = st.sidebar.slider("Time Index", 0, len(times) - 1, 0)
        st.sidebar.write(f"Selected time: {times[time_index]}")

        # Grid and Data
        lon2d, lat2d = generate_latlon_grid_from_griddesc()
        data = ds[variable].isel(TSTEP=time_index, LAY=0)

        # --- Plot with Cartopy ---
        st.subheader(f"{variable} at {times[time_index]} (Layer 1)")
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        mesh = ax.pcolormesh(lon2d, lat2d, data.values, transform=ccrs.PlateCarree(), cmap='viridis')
        # --- Load WRF wind data ---
        import glob

        wrf_files = sorted(glob.glob("/home/duch/cmaq/hanoijan/WRF/2022-01-25/d01/wrfout_d01_2022-01-25_*"))  # Update this path!
        wrf_ds = xr.open_mfdataset(wrf_files, combine='by_coords')

        # --- Match CMAQ time to WRF time ---
        cmaq_time = times[time_index]
        wrf_time_idx = np.argmin(np.abs(wrf_ds["XTIME"].values - np.datetime64(cmaq_time)))

        # --- Extract wind and coordinates ---
        u10 = wrf_ds["U10"].isel(Time=wrf_time_idx).values
        v10 = wrf_ds["V10"].isel(Time=wrf_time_idx).values
        lats = wrf_ds["XLAT"].isel(Time=wrf_time_idx).values
        lons = wrf_ds["XLONG"].isel(Time=wrf_time_idx).values

        # --- Downsample to reduce clutter ---
        step = 2
        ax.quiver(lons[::step, ::step], lats[::step, ::step], u10[::step, ::step], v10[::step, ::step],
                  transform=ccrs.PlateCarree(), scale=700, width=0.002, color="black")


        plt.colorbar(mesh, ax=ax, label=ds[variable].units)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(f"{variable} at {times[time_index]}")
        st.pyplot(fig)

        # --- Folium map for interaction ---
        st.subheader("Click on the map to see time series")
        center_lat = np.mean(lat2d)
        center_lon = np.mean(lon2d)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        # Display map
        output = st_folium(fmap, height=400, width=700)

        # --- Time Series on Click ---
        if output and output.get("last_clicked"):
            click_lat = output["last_clicked"]["lat"]
            click_lon = output["last_clicked"]["lng"]

            st.info(f"Clicked at lat={click_lat:.3f}, lon={click_lon:.3f}")

            # Find nearest grid point
            flat_lats = lat2d.flatten()
            flat_lons = lon2d.flatten()
            distances = np.sqrt((flat_lats - click_lat)**2 + (flat_lons - click_lon)**2)
            idx = np.argmin(distances)
            ny, nx = lat2d.shape
            i, j = np.unravel_index(idx, (ny, nx))

            st.write(f"Nearest grid index: ({i}, {j})")

            # Extract time series
            series = ds[variable][:, 0, i, j].values
            df_ts = pd.DataFrame({'Time': times, variable: series})

            # Plot time series
            fig2, ax2 = plt.subplots()
            ax2.plot(df_ts['Time'], df_ts[variable], label=f"{variable} at ({i},{j})")
            ax2.set_xlabel("Time")
            ax2.set_ylabel(f"{variable} ({ds[variable].units})")
            ax2.set_title(f"Time Series at lat={click_lat:.2f}, lon={click_lon:.2f}")
            ax2.grid(True)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Failed to load or plot file: {e}")

