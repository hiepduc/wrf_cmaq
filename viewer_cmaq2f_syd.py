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
import glob
import time


# --- Function to generate lat/lon grid from GRIDDESC ---
def generate_latlon_grid_from_griddesc():
    # Projection info from GRIDDESC
    lat1 = -34.0
    lat2 = -22.0
    central_lon = 134.0
    center_lat = -28.0  # Projection origin
    x0 = 1490000.25
    y0 = -842000.312
    dx = 2000.0
    dy = 2000.0
    nx = 88
    ny = 88

    # Create grid in projected coordinates
    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(x, y)

    # Lambert Conformal projection (same as in WRF)
    lambert_proj = Proj(proj="lcc", lat_1=lat1, lat_2=lat2, lat_0=center_lat,
                        lon_0=central_lon, ellps="WGS84")

    # Convert to lat/lon
    transformer = Transformer.from_proj(lambert_proj, "epsg:4326", always_xy=True)
    lon2d, lat2d = transformer.transform(X, Y)

    return lon2d, lat2d

def plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind):
    data = ds[variable].isel(TSTEP=time_index, LAY=0)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    mesh = ax.pcolormesh(lon2d, lat2d, data.values, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto', zorder=1)
    plt.colorbar(mesh, ax=ax, label=ds[variable].units)

    if show_wind:
        try:
            wrf_times_raw = wrf_ds['Times'].values
            wrf_times = pd.to_datetime(["".join(t.astype(str)).replace("_", " ").strip() for t in wrf_times_raw])
            wrf_time_idx = np.argmin(np.abs(wrf_times - pd.to_datetime(cmaq_time)))
            wrf_sel = wrf_ds.isel(Time=wrf_time_idx).load()

            uu = wrf_sel["UU"].isel(num_metgrid_levels=0).values
            vv = wrf_sel["VV"].isel(num_metgrid_levels=0).values

            # Unstagger U and trim to 88x88
            u_unstaggered = 0.5 * (uu[:, :-1] + uu[:, 1:])
            v_unstaggered = 0.5 * (vv[:-1, :] + vv[1:, :])

            # Trim or crop to match the 88x88 CMAQ grid
            u = u_unstaggered[:88, :88]
            v = v_unstaggered[:88, :88]

            # Generate lat/lon grid from GRIDDESC
            # You already have:
            # lon2d, lat2d = generate_latlon_grid_from_griddesc()

            step = 4
            ax.quiver(
                lon2d[::step, ::step],
                lat2d[::step, ::step],
                u[::step, ::step],
                v[::step, ::step],
                transform=ccrs.PlateCarree(),
                scale=50,
                width=0.002,
                color="white",
                alpha=0.7,
                zorder=2
            )

        except Exception as e:
            st.warning(f"Could not overlay wind vectors: {e}")

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"{variable} + Wind Vectors at {cmaq_time}")
    return fig


# --- Streamlit app ---
st.set_page_config(layout="wide")
st.title("CMAQ Concentration Viewer with Time Series")

nc_path = st.sidebar.text_input("Path to CMAQ NetCDF file:",
    "/mnt/scratch_lustre/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/CTM/2024-02-10/d03/CCTM_ACONC_v532_intel_d03_20240210.nc")

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

        # In the Streamlit sidebar
        show_wind = st.sidebar.checkbox("Show Wind Overlay", value=True)
        animate = st.sidebar.checkbox("Play Animation")

        # Generate lon/lat grid once
        lon2d, lat2d = generate_latlon_grid_from_griddesc()

        # Load WRF files once for animation or static
        wrf_files = sorted(glob.glob("/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/WRF/run/met_em.d03.2024-02-10_*"))
        wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time")

        # Animation mode
        import os
        import imageio
        from io import BytesIO

        # New UI: Add record option
        record = st.sidebar.checkbox("Record Animation to MP4")

        if animate:
            placeholder = st.empty()
            frames = []  # store images for video
            for time_index in range(len(times)):
                cmaq_time = times[time_index]
                with placeholder.container():
                    fig = plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind)
                    st.pyplot(fig)

                    if record:
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150)
                        buf.seek(0)
                        frames.append(imageio.v3.imread(buf.getvalue()))
                time.sleep(0.5)  # Controls speed
            # Save video
            if record and frames:
                output_path = "/tmp/cmaq_animation.mp4"
                imageio.mimsave(output_path, frames, fps=2)
                with open(output_path, "rb") as f:
                    st.download_button("🎥 Download MP4", f, file_name="cmaq_animation.mp4")


        else:
            time_index = st.sidebar.slider("Select time index", 0, len(times) - 1, 0)
            cmaq_time = times[time_index]
            fig = plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind)
            st.pyplot(fig)

        st.sidebar.write(f"Selected time: {times[time_index]}")

        # --- Folium map for interaction ---
        st.subheader("Click on the map to see time series")
        center_lat = np.mean(lat2d)
        center_lon = np.mean(lon2d)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        stations = [
            {"name": "Parramatta", "lat": -33.797, "lon": 151.002},
            {"name": "Richmond", "lat": -33.60, "lon": 150.7514},
            {"name": "Rouse Hill", "lat": -33.68, "lon": 150.92},
            {"name": "Bargo", "lat": -34.30, "lon": 150.57},
            {"name": "Campbelltown", "lat": -34.07, "lon": 150.82},
            {"name": "Liverpool", "lat": -33.92, "lon": 150.923},
            {"name": "Bringelly", "lat": -33.93, "lon": 150.73},
            {"name": "Singleton", "lat": -32.57, "lon": 151.178},
            {"name": "Muswellbrook", "lat": -32.261, "lon": 150.89},
            {"name": "Merriwa", "lat": -32.139, "lon": 150.356},
            {"name": "Albion Park South", "lat": -34.567, "lon": 150.80},
            {"name": "Kembla Grange", "lat": -34.47, "lon": 150.796},
            {"name": "Wollongong", "lat": -34.424, "lon": 150.893}
        ]

        for station in stations:
            folium.Marker(
                location=[station["lat"], station["lon"]],
                popup=station["name"],
                icon=folium.Icon(color="blue", icon="cloud"),
            ).add_to(fmap)

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

