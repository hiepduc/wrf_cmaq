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
import os
import requests
from datetime import datetime, timedelta

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

def load_day_domain_nc(nc_path, variable, lat_target, lon_target):
    try:
        ds = xr.open_dataset(nc_path)

        # Find nearest lat/lon grid
        lats = ds['LAT'].values if 'LAT' in ds else ds['lat'].values
        lons = ds['LON'].values if 'LON' in ds else ds['lon'].values
        dist = np.sqrt((lats - lat_target)**2 + (lons - lon_target)**2)
        i_lat, i_lon = np.unravel_index(np.argmin(dist), dist.shape)

        ts = ds[variable][:, 0, i_lat, i_lon]
        times = pd.to_datetime(ds['time'].values)

        return pd.DataFrame({'Time': times, variable: ts.values})
    except Exception as e:
        print(f"❌ Failed to load {nc_path.name}: {e}")
        return None

def build_timeseries_across_days(base_dir, variable, lat, lon, domain="d03"):
    all_dfs = []

    base = Path(base_dir)
    for date_dir in sorted(base.glob("20*/")):
        domain_dir = date_dir / domain
        nc_files = list(domain_dir.glob("CCTM_ACONC_*.nc"))

        if not nc_files:
            continue

        for nc_file in nc_files:
            df = load_day_domain_nc(nc_file, variable, lat, lon)
            if df is not None:
                all_dfs.append(df)

    if all_dfs:
        df_full = pd.concat(all_dfs).drop_duplicates(subset="Time").sort_values("Time")
        return df_full.set_index("Time")
    else:
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def fetch_observation_data(station_name, parameter_code, start_date_str, end_date_str):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    # Get site and parameter mappings
    site_map = {s["SiteName"]: s["Site_Id"] for s in requests.get(sites_url, headers=HEADERS).json()}
    param_map = {
        p["ParameterCode"]: p["ParameterCode"]
        for p in requests.get(params_url, headers=HEADERS).json()
        if "hour" in p.get("Frequency", "").lower()
    }

    if station_name not in site_map:
        st.error(f"Station '{station_name}' not found in API.")
        return None
    if parameter_code not in param_map:
        st.error(f"Parameter '{parameter_code}' not available for hourly data.")
        return None

    site_id = site_map[station_name]
    parameter_id = param_map[parameter_code]

    payload = {
        "Sites": [site_id],
        "Parameters": [parameter_id],
        "StartDate": start_date_str,
        "EndDate": end_date_str,
        "Categories": ["Averages"],
        "SubCategories": ["Hourly"],
        "Frequency": ["Hourly average"]
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60).json()
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch data from API: {e}")
        return None

    records = [
        {
            "datetime": datetime.strptime(r["Date"], "%Y-%m-%d") + timedelta(hours=r["Hour"]),
            "obs": r["Value"]
        }
        for r in resp if r["Value"] is not None
    ]

    if not records:
        st.warning(f"No valid data returned from API for {station_name}")
        return None

    df_obs = pd.DataFrame(records).set_index("datetime").sort_index()
    return df_obs

from sklearn.metrics import mean_squared_error

def compute_metrics(df):
    obs = df["obs"]
    pred = df["pred"]

    correlation = obs.corr(pred)
    rmse = mean_squared_error(obs, pred, squared=False)
    me = (pred - obs).mean()
    mb = ((pred - obs) / obs.replace(0, np.nan)).mean() * 100

    return {
        "Correlation": correlation,
        "RMSE": rmse,
        "Mean Error (ME)": me,
        "Mean Bias (MB%)": mb
    }


# --- Function to generate lat/lon grid from GRIDDESC ---
def generate_latlon_grid_from_griddesc():
    # Projection info from GRIDDESC
    lat1 = -34.0
    lat2 = -22.0
    central_lon = 134.0
    center_lat = -28.0  # Projection origin
    x0 = 1169998.875 
    y0 = -1133998.750
    dx = 4000.0
    dy = 4000.0
    nx = 180
    ny = 237

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

            # Unstagger U and trim to 189x237
            u_unstaggered = 0.5 * (uu[:, :-1] + uu[:, 1:])
            v_unstaggered = 0.5 * (vv[:-1, :] + vv[1:, :])

            # Trim or crop to match the 180x237 CMAQ grid
            u = u_unstaggered[:237, :180]
            v = v_unstaggered[:237, :180]

            # Generate lat/lon grid from GRIDDESC
            # You already have:
            # lon2d, lat2d = generate_latlon_grid_from_griddesc()

            step = 10
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

nc_path = st.sidebar.text_input("Path to CMAQ base directory of daily NetCDF file:",
    "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/CTM")

# Base directory containing daily folders of CMAQ output
base_dir = "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/CTM"

# Base directory containing WRF meteorological files
base_dir_wrf = "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/WRF/run/met_em.d02.*.nc"

# Get all d02 NetCDF files across date subdirectories
file_list = sorted(glob.glob(os.path.join(base_dir, "2024-*/d02/CCTM_ACONC_v532_intel_d02*.nc")))

# Debug print (optional)
print(f"Found {len(file_list)} files:")
for f in file_list:
    print("  ", f)


if nc_path:
    try:
        # Combine all files into one Dataset
        ds = xr.open_mfdataset(
            file_list,
            combine="nested",
            concat_dim="TSTEP",
            #compat="override",  # override metadata conflict
            parallel=True
        )

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
        wrf_files = sorted(glob.glob(base_dir_wrf))
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
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        stations = [
            {"name": "PARRAMATTA NORTH", "lat": -33.797, "lon": 151.002},
            {"name": "RICHMOND", "lat": -33.60, "lon": 150.7514},
            {"name": "ROUSE HILL", "lat": -33.68, "lon": 150.92},
            {"name": "BARGO", "lat": -34.30, "lon": 150.57},
            {"name": "CAMPBELLTOWN WEST", "lat": -34.07, "lon": 150.82},
            {"name": "LIVERPOOL", "lat": -33.92, "lon": 150.923},
            {"name": "BRINGELLY", "lat": -33.93, "lon": 150.73},
            {"name": "SINGLETON", "lat": -32.57, "lon": 151.178},
            {"name": "MUSWELLBROOK", "lat": -32.261, "lon": 150.89},
            {"name": "MERRIWA", "lat": -32.139, "lon": 150.356},
            {"name": "ALBION PARK SOUTH", "lat": -34.567, "lon": 150.80},
            {"name": "KEMBLA GRANGE", "lat": -34.47, "lon": 150.796},
            {"name": "WOLLONGONG", "lat": -34.424, "lon": 150.893}
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

        # This section is for validation with observed data from API
        # Station and pollutant parameter are selected from the lists on sideboard.
        # The date is also selected from sideboard
        # Then get the prediction from WRF-CMAQ 

        # Mapping model parameter names to API parameter names
        PARAMETER_API_MAP = {
            "O3": "OZONE",
            "PM2.5": "PM25",
            "PM10": "PM10",
            "NO2": "NO2",
            # Add more if needed
        }

        # --- Validation Section ---
        st.subheader("📡 Model vs Observation Validation")

        # Sidebar selections
        station_name = st.sidebar.selectbox("Select Station for Validation", [s["name"] for s in stations])
        parameter_code = st.sidebar.selectbox("Select Parameter", ["NO2", "O3", "PM10", "PM2.5"])  # Customize as needed
        # Use available model time as fallback for initial range
        #default_start = pd.to_datetime(times.min())
        #default_end = pd.to_datetime(times.max())
        default_start = min(times)
        default_end = max(times)

        date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])

        # Find station info by name
        station_info = next((s for s in stations if s["name"] == station_name), None)

        if station_info and len(date_range) == 2:
            station_lat = station_info["lat"]
            station_lon = station_info["lon"]

            start_date, end_date = date_range
            # continue with processing...
            # Get index of nearest grid point
            # Flatten the lat/lon grids
            flat_lats = lat2d.flatten()
            flat_lons = lon2d.flatten()

            # Compute distance to station
            distances = np.sqrt((flat_lats - station_lat)**2 + (flat_lons - station_lon)**2)
            min_idx = np.argmin(distances)

            # Get 2D grid indices
            ny, nx = lat2d.shape
            i, j = np.unravel_index(min_idx, (ny, nx))

            series = ds[parameter_code][:, 0, i, j].values
            df_ts = pd.DataFrame({'Time': times, 'pred': series})

            # Model predictions
            df_pred = df_ts.set_index("Time")
            df_pred = df_pred[(df_pred.index >= pd.to_datetime(start_date)) & (df_pred.index <= pd.to_datetime(end_date))]

            # Convert model UTC time to Australia/Sydney local time
            df_pred.index = df_pred.index.tz_localize("UTC").tz_convert("Australia/Sydney")
            df_pred.index.name = "datetime_local"

            # --- Fetch observation ---
            # Map model parameter to API parameter
            api_param_code = PARAMETER_API_MAP.get(parameter_code, parameter_code)
            df_obs = fetch_observation_data(
                station_name=station_name,
                parameter_code=api_param_code,
                start_date_str=start_date.strftime("%Y-%m-%d"),
                end_date_str=end_date.strftime("%Y-%m-%d")
            )

            df_obs.index = pd.to_datetime(df_obs.index).tz_localize("Australia/Sydney")

            # Ensure obs index is also named datetime_local
            df_obs.index.name = "datetime_local"

            # Merge and convert observed units from pphm to ppm if needed
            df_merged = pd.merge(df_pred, df_obs / 100, left_index=True, right_index=True, how='inner')

            print(f"Observed shape: {df_obs.shape}, Predicted shape: {df_pred.shape}")

            if not df_merged.empty:
                st.markdown(f"### 📊 Comparison at **{station_name}**")

                fig3, ax3 = plt.subplots()
                ax3.plot(df_merged.index, df_merged["pred"], label="CMAQ Prediction")
                ax3.plot(df_merged.index, df_merged["obs"], label="Observation", linestyle='--')
                ax3.set_title(f"{parameter_code} Concentration")
                ax3.set_ylabel(f"{parameter_code} ({ds[parameter_code].units})")
                ax3.set_xlabel("Time")
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)

                # --- Compute validation metrics ---
                st.line_chart(df_merged.rename(columns={"pred": "Model", "obs": "Observed"}))

                from sklearn.metrics import mean_squared_error
                rmse = mean_squared_error(df_merged["obs"], df_merged["pred"], squared=False)
                st.metric("RMSE", f"{rmse:.3f} ppm")

                metrics = compute_metrics(df_merged)
                st.markdown("### 📈 Validation Metrics")
                st.write(metrics)
            else:
                st.warning("No matching time data found between prediction and observation.")

    except Exception as e:
        st.error(f"Failed to load or plot file: {e}")

