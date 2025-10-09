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
def generate_latlon_grid_from_griddesc(griddesc_file):
    """
    Read MCIP GRIDDESC and return lon2d, lat2d and grid_info dict.
    grid_info contains: xcell (m), ycell (m), nx, ny, origin_x, origin_y
    """
    import numpy as np
    from pyproj import Proj, Transformer

    # read and strip quotes / empty lines
    with open(griddesc_file, "r") as f:
        lines = [ln.strip().replace("'", "") for ln in f if ln.strip()]

    # 1) find the *grid definition* line: looks like
    #    <proj_name>  origin_x origin_y xcell ycell nx ny ...
    proj_name = None
    origin_x = origin_y = xcell = ycell = nx = ny = None
    for ln in lines:
        toks = ln.split()
        if len(toks) >= 7:
            # first token is non-numeric (projection name), rest mostly numeric
            try:
                # try parse tokens[1:] as floats -> if many succeed it's the grid line
                nums = [float(t) for t in toks[1:] ]
                if len(nums) >= 6:
                    proj_name = toks[0]
                    origin_x, origin_y, xcell, ycell = nums[0], nums[1], nums[2], nums[3]
                    nx, ny = int(round(nums[4])), int(round(nums[5]))
                    break
            except Exception:
                continue

    if proj_name is None:
        raise ValueError(f"Could not find grid-definition line in GRIDDESC: {griddesc_file}")

    # 2) find projection parameters for proj_name: look for a separate line equal to proj_name
    #    followed by a numeric line like: 2 lat1 lat2 lon1 lon2 lat0
    lat1 = lat2 = lon0 = lat0 = None
    for i, ln in enumerate(lines):
        if ln == proj_name:
            # next line should have projection params
            if i + 1 < len(lines):
                toks = lines[i + 1].split()
                # try parse numeric tokens
                nums = []
                for t in toks:
                    try:
                        nums.append(float(t))
                    except Exception:
                        pass
                # expected pattern: [code, lat1, lat2, lon1, lon2, lat0] (based on examples)
                if len(nums) >= 6:
                    lat1 = nums[1]
                    lat2 = nums[2]
                    lon0 = nums[3]   # usually the central meridian
                    lat0 = nums[5]
                    break

    if any(v is None for v in (lat1, lat2, lon0, lat0)):
        # fallback: try to find any line with 6 numeric tokens and use that
        for ln in lines:
            toks = ln.split()
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except Exception:
                    pass
            if len(nums) >= 6:
                lat1, lat2, lon0, lat0 = nums[1], nums[2], nums[3], nums[5]
                break

    if any(v is None for v in (lat1, lat2, lon0, lat0)):
        raise ValueError("Could not determine projection parameters (lat1, lat2, lon0, lat0) from GRIDDESC.")

    # 3) build projected grid (cell centers)
    x = origin_x + xcell * (np.arange(nx) + 0.5)
    y = origin_y + ycell * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(x, y)   # shape (ny, nx)

    # 4) transform to lat/lon using Lambert Conformal Conic
    lambert_proj = Proj(proj="lcc", lat_1=lat1, lat_2=lat2, lat_0=lat0,
                        lon_0=lon0, ellps="WGS84")
    transformer = Transformer.from_proj(lambert_proj, "epsg:4326", always_xy=True)
    lon2d, lat2d = transformer.transform(X, Y)

    grid_info = {
        "proj_name": proj_name,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "xcell": xcell,
        "ycell": ycell,
        "nx": nx,
        "ny": ny,
        "lat1": lat1,
        "lat2": lat2,
        "lon0": lon0,
        "lat0": lat0
    }

    return lon2d, lat2d, grid_info

import fnmatch
import os

import fnmatch
import os

def find_apmdiag_for_aconc(aconc_path):
    """Given path to a CCTM_ACONC file or to a directory containing them,
    try to find the corresponding CCT_APMDIAG* file in the same directory."""
    
    if os.path.isdir(aconc_path):
        base_dir = aconc_path
    else:
        base_dir = os.path.dirname(aconc_path)
    
    # Debugging: Print the directory we are searching
    print(f"Looking for APMDIAG in: {base_dir}")
    
    # Look for files named like CCT_APMDIAG*.nc in that directory
    for fname in sorted(os.listdir(base_dir)):
        print(f"Checking file: {fname}")  # Debugging: Print file names found in the directory
        if fname.startswith("CCTM_APMDIAG"):  # Match the prefix directly
            print(f"Found APMDIAG file: {fname}")  # Debugging: Found the correct file
            return os.path.join(base_dir, fname)
    
    # Debugging: If no file is found
    print("APMDIAG file not found!")
    return None

def compute_pm25_from_aconc_apmdiag_for_all_days(ds_aconc, ds_apmdiag):
    """Compute PM2.5 for all days using the formula and available data."""
    # Initialize an array to store PM2.5 for all days
    pm25_all_days = []

    # Loop through all time steps (TSTEP)
    for t in range(ds_aconc.sizes['TSTEP']):
        # Extract relevant variables from ACONC for the current time step
        aso4i = ds_aconc["ASO4I"].isel(TSTEP=t).values
        ano3i = ds_aconc["ANO3I"].isel(TSTEP=t).values
        anh4i = ds_aconc["ANH4I"].isel(TSTEP=t).values
        # More variables (add the ones in your formula)
        anai = ds_aconc["ANAI"].isel(TSTEP=t).values
        acli = ds_aconc["ACLI"].isel(TSTEP=t).values
        aeci = ds_aconc["AECI"].isel(TSTEP=t).values
        alvoo1i = ds_aconc["ALVOO1I"].isel(TSTEP=t).values
        alvoo2i = ds_aconc["ALVOO2I"].isel(TSTEP=t).values
        asvoo1i = ds_aconc["ASVOO1I"].isel(TSTEP=t).values
        asvoo2i = ds_aconc["ASVOO2I"].isel(TSTEP=t).values
        alvpo1i = ds_aconc["ALVPO1I"].isel(TSTEP=t).values
        asvpo1i = ds_aconc["ASVPO1I"].isel(TSTEP=t).values
        asvpo2i = ds_aconc["ASVPO2I"].isel(TSTEP=t).values
        aothri = ds_aconc["AOTHRI"].isel(TSTEP=t).values
        aothri = ds_aconc["AOTHRI"].isel(TSTEP=t).values

        aso4j = ds_aconc["ASO4J"].isel(TSTEP=t).values
        ano3j = ds_aconc["ANO3J"].isel(TSTEP=t).values
        anh4j = ds_aconc["ANH4J"].isel(TSTEP=t).values
        anaj = ds_aconc["ANAJ"].isel(TSTEP=t).values
        aclj = ds_aconc["ACLJ"].isel(TSTEP=t).values
        aecj = ds_aconc["AECJ"].isel(TSTEP=t).values
        #axyl1j = ds_aconc["AXYL1J"].isel(TSTEP=t).values
        #axyl2j = ds_aconc["AXYL2J"].isel(TSTEP=t).values
        #axyl3j = ds_aconc["AXYL3J"].isel(TSTEP=t).values
        #atol1j = ds_aconc["ATOL1J"].isel(TSTEP=t).values
        #atol2j = ds_aconc["ATOL2J"].isel(TSTEP=t).values
        #atol3j = ds_aconc["ATOL3J"].isel(TSTEP=t).values
        #abnz1j = ds_aconc["ABNZ1J"].isel(TSTEP=t).values
        #abnz2j = ds_aconc["ABNZ2J"].isel(TSTEP=t).values
        #abnz3j = ds_aconc["ABNZ3J"].isel(TSTEP=t).values
        aiso1j = ds_aconc["AISO1J"].isel(TSTEP=t).values
        aiso2j = ds_aconc["AISO2J"].isel(TSTEP=t).values
        aiso3j = ds_aconc["AISO3J"].isel(TSTEP=t).values
        #atrp1j = ds_aconc["ATRP1J"].isel(TSTEP=t).values
        #atrp2j = ds_aconc["ATRP2J"].isel(TSTEP=t).values
        #atrp3j = ds_aconc["ATRP3J"].isel(TSTEP=t).values
        #aalk1j = ds_aconc["AALK1J"].isel(TSTEP=t).values
        #aalk2j = ds_aconc["AALK2J"].isel(TSTEP=t).values
        #apah1j = ds_aconc["APAH1J"].isel(TSTEP=t).values
        #apah2j = ds_aconc["APAH2J"].isel(TSTEP=t).values
        #apah3j = ds_aconc["APAH3J"].isel(TSTEP=t).values
        aorgcj = ds_aconc["AORGCJ"].isel(TSTEP=t).values
        aolgbj = ds_aconc["AOLGBJ"].isel(TSTEP=t).values
        aolgaj = ds_aconc["AOLGAJ"].isel(TSTEP=t).values
        alvoo1j = ds_aconc["ALVOO1J"].isel(TSTEP=t).values
        alvoo2j = ds_aconc["ALVOO2J"].isel(TSTEP=t).values
        asvoo1j = ds_aconc["ASVOO1J"].isel(TSTEP=t).values
        asvoo2j = ds_aconc["ASVOO2J"].isel(TSTEP=t).values
        asvoo3j = ds_aconc["ASVOO3J"].isel(TSTEP=t).values
        apcsoj = ds_aconc["APCSOJ"].isel(TSTEP=t).values
        alvpo1j = ds_aconc["ALVPO1J"].isel(TSTEP=t).values
        asvpo1j = ds_aconc["ASVPO1J"].isel(TSTEP=t).values
        asvpo2j = ds_aconc["ASVPO2J"].isel(TSTEP=t).values
        asvpo3j = ds_aconc["ASVPO3J"].isel(TSTEP=t).values
        aivpo1j = ds_aconc["AIVPO1J"].isel(TSTEP=t).values
        aothrj = ds_aconc["AOTHRJ"].isel(TSTEP=t).values
        afej = ds_aconc["AFEJ"].isel(TSTEP=t).values
        asij = ds_aconc["ASIJ"].isel(TSTEP=t).values
        atij = ds_aconc["ATIJ"].isel(TSTEP=t).values
        acij = ds_aconc["ACAJ"].isel(TSTEP=t).values
        amgj = ds_aconc["AMGJ"].isel(TSTEP=t).values
        amnj = ds_aconc["AMNJ"].isel(TSTEP=t).values
        aalj = ds_aconc["AALJ"].isel(TSTEP=t).values
        akj = ds_aconc["AKJ"].isel(TSTEP=t).values

        asoil = ds_aconc["ASOIL"].isel(TSTEP=t).values
        acors = ds_aconc["ACORS"].isel(TSTEP=t).values
        aseacat = ds_aconc["ASEACAT"].isel(TSTEP=t).values
        aclk = ds_aconc["ACLK"].isel(TSTEP=t).values
        aso4k = ds_aconc["ASO4K"].isel(TSTEP=t).values
        ano3k = ds_aconc["ANO3K"].isel(TSTEP=t).values
        anh4k = ds_aconc["ANH4K"].isel(TSTEP=t).values
        
        # Extract PM2.5 fractions from APMDIAG for the current time step
        pm25at = ds_apmdiag["PM25AT"].isel(TSTEP=t).values
        pm25ac = ds_apmdiag["PM25AC"].isel(TSTEP=t).values
        pm25co = ds_apmdiag["PM25CO"].isel(TSTEP=t).values
        
        # Calculate PM2.5 for the current time step using the formula
        pm25 = (aso4i + ano3i + anh4i + anai + acli + aeci + alvoo1i + alvoo2i + asvoo1i + asvoo2i + alvpo1i + asvpo1i + asvpo2i + aothri) * pm25at + (aso4j + ano3j + anh4j + anaj + aclj + aecj +  aiso1j + aiso2j + aiso3j + aorgcj + aolgbj + aolgaj + alvoo1j + alvoo2j + asvoo1j + asvoo2j + asvoo3j + apcsoj + alvpo1j + asvpo1j + asvpo2j + asvpo3j + aivpo1j + aothrj + afej + asij + atij + acij + amgj + amnj + aalj + akj) * pm25ac + (asoil + acors + aseacat + aclk + aso4k + ano3k + anh4k) * pm25co
        
        # Append the PM2.5 values for this time step to the list
        pm25_all_days.append(pm25)

    # Convert the list to a numpy array and stack it as a dataset
    pm25_all_days = np.stack(pm25_all_days, axis=0)  # Shape (TSTEP, LAY, ROW, COL)

    return pm25_all_days


# Adjust the step size for plotting wind vector density based on domain size (nx, ny)
def calculate_step_size(nx, ny):
    # Simple logic to reduce step size for smaller grids
    if nx < 90 and ny < 90:
        return 8  # Increase step size (fewer arrows) for smaller domains
    elif nx < 200 and ny < 350:
        return 6  # A moderate step for mid-sized domains
    else:
        return 20  # Default step for larger domains like d01

def plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind, nx, ny):
    data = ds[variable].isel(TSTEP=time_index, LAY=0)
    ############
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

            # Unstagger U and trim or interpolate to match CMAQ grid
            u_unstaggered = 0.5 * (uu[:, :-1] + uu[:, 1:])
            v_unstaggered = 0.5 * (vv[:-1, :] + vv[1:, :])

            # Trim or interpolate to the CMAQ grid resolution (nx, ny)
            u_trimmed = u_unstaggered[:ny, :nx]
            v_trimmed = v_unstaggered[:ny, :nx]

            # If the WRF grid is smaller than CMAQ, we need to interpolate
            if u_trimmed.shape != (ny, nx):
                from scipy.interpolate import interp2d
                f_u = interp2d(np.arange(u_unstaggered.shape[1]), np.arange(u_unstaggered.shape[0]), u_unstaggered)
                f_v = interp2d(np.arange(v_unstaggered.shape[1]), np.arange(v_unstaggered.shape[0]), v_unstaggered)
                u_trimmed = f_u(np.linspace(0, u_unstaggered.shape[1] - 1, nx), np.linspace(0, u_unstaggered.shape[0] - 1, ny))
                v_trimmed = f_v(np.linspace(0, v_unstaggered.shape[1] - 1, nx), np.linspace(0, v_unstaggered.shape[0] - 1, ny))

            # Generate lat/lon grid from GRIDDESC (use the previously calculated lon2d, lat2d)
            step = calculate_step_size(nx, ny)  # Calculate the step based on grid size
            ax.quiver(
                lon2d[::step, ::step],
                lat2d[::step, ::step],
                u_trimmed[::step, ::step],
                v_trimmed[::step, ::step],
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

# --- Helper: Read GRIDDESC ---
import re

def read_griddesc(griddesc_file):
    """
    Robustly parse GRIDDESC. Returns dict with xcell (m), ycell (m), nx, ny
    or None if not found/parsable.
    """
    with open(griddesc_file, "r") as f:
        lines = [ln.strip().replace("'", "") for ln in f if ln.strip()]

    for line in lines:
        # split into tokens
        tokens = line.split()
        # try to convert tokens to floats where possible
        floats = []
        for tok in tokens:
            # allow things like -2190000.000 or 12000.000
            try:
                v = float(tok)
                floats.append(v)
            except Exception:
                # skip non-numeric tokens
                continue

        # we expect at least 6-7 numeric fields in the interesting line
        # pattern seen: [x0, y0, xcell, ycell, nx, ny, something]
        if len(floats) >= 6:
            # pick the slice that matches typical GRIDDESC layout:
            # use the last 5..2 numeric tokens: [ ..., xcell, ycell, nx, ny, ... ]
            # we'll choose floats[-5], floats[-4], floats[-3], floats[-2]
            try:
                xcell = float(floats[-5])
                ycell = float(floats[-4])
                nx = int(round(floats[-3]))
                ny = int(round(floats[-2]))

                # basic sanity checks
                if xcell > 0 and ycell > 0 and nx > 0 and ny > 0:
                    return {"xcell": xcell, "ycell": ycell, "nx": nx, "ny": ny}
            except Exception:
                # if any conversion fails, skip this line
                continue

    return None


# --- Streamlit app ---
st.set_page_config(layout="wide")
st.title("CMAQ Concentration Viewer with Time Series")

# --- Sidebar ---
st.sidebar.header("Configuration")

# Domain selection
domain = st.sidebar.selectbox("Select CMAQ Domain", ["d01", "d02", "d03"])

# Paths
cmaq_base = st.sidebar.text_input(
    "Path to CMAQ base directory of daily NetCDF files:",
    "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/CTM"
)

mcip_base = st.sidebar.text_input(
    "Path to MCIP base directory:",
    "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/MCIP"
)

wrf_base = st.sidebar.text_input(
    "Path to WRF meteorological base directory:",
    "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/WRF/run"
)

# --- Find GRIDDESC file for chosen domain ---
date_dirs = sorted(glob.glob(os.path.join(mcip_base, "2024-*")))
if not date_dirs:
    st.error(f"No date directories found in {mcip_base}")
else:
    # Pick first date folder
    first_date = date_dirs[0]
    griddesc_file = os.path.join(first_date, domain, "GRIDDESC")
    if os.path.exists(griddesc_file):
        grid_info = read_griddesc(griddesc_file)
        if grid_info:
            st.sidebar.write(f"Grid resolution: {grid_info['xcell']/1000:.1f} km")
            st.sidebar.write(f"Grid size: {grid_info['nx']} x {grid_info['ny']}")
        else:
            st.error(f"Could not parse grid info from {griddesc_file}")
    else:
        st.error(f"GRIDDESC not found: {griddesc_file}")

# --- File discovery ---
if os.path.exists(cmaq_base):
    file_list = sorted(glob.glob(
        os.path.join(cmaq_base, f"2024-*/{domain}/CCTM_ACONC_v532_intel_{domain}_*.nc")
    ))
    st.sidebar.write(f"Found {len(file_list)} CMAQ files.")

    wrf_files = sorted(glob.glob(
        os.path.join(wrf_base, f"met_em.{domain}.*.nc")
    ))
    st.sidebar.write(f"Found {len(wrf_files)} WRF files.")
else:
    st.warning("Please check your CMAQ base directory path.")

# Now you can continue your existing code using `file_list` and `wrf_files`

#if nc_path:
if file_list:
    try:
        valid_files = []
        for f in file_list:
            try:
                #with xr.open_dataset(f, engine="pseudonetcdf") as test_ds:
                with xr.open_dataset(f) as test_ds:
                    _ = test_ds.variables  # force load metadata
                valid_files.append(f)
                print(f"OK: {f}")
            except Exception as e:
                print(f"Skipping {f}: {e}")

        if not valid_files:
            raise RuntimeError("No valid CMAQ files found!")

        # Now combine only valid files
        ds = xr.open_mfdataset(
            valid_files,
            combine="nested",
            concat_dim="TSTEP",
            #engine="netcdf4",
            parallel=True
        ).isel(LAY=[0])

        # Select only the first layer (LAY=0)
        #ds = ds.isel(LAY=0)
        st.sidebar.success("CMAQ file loaded.")

        # --- Sidebar for variable selection ---
        variables = [v for v in ds.data_vars if v != "TFLAG"] + ['PM2.5']  # Add PM2.5 as an option
        variable = st.sidebar.selectbox("Select variable", variables)

        if variable.lower() in ("pm2.5", "pm25", "pm_2_5", "pm-2.5"):
            # Find all ACONC and APMDIAG files
            all_aconc_files = file_list if file_list else None
            all_apm_files = []

            for aconc_file in all_aconc_files:
                # Find the corresponding APMDIAG file for each ACONC file
                apm_file = find_apmdiag_for_aconc(aconc_file)
                if apm_file and os.path.exists(apm_file):
                    all_apm_files.append(apm_file)
                else:
                    st.warning(f"APMDIAG file not found for {aconc_file}; cannot compute PM2.5.")

            if not all_apm_files:
                st.warning("No valid APMDIAG files found for the given ACONC files.")
            else:
                # Debugging: Show the paths of the APMDIAG files
                print(f"Found APMDIAG files: {all_apm_files}")
        
                # Load and process APMDIAG files
                ds_apmdiag_list = []
                for apm_file in all_apm_files:
                    ds_apmdiag_list.append(xr.open_dataset(apm_file))

                # Now calculate PM2.5 for all days from all ACONC and APMDIAG datasets
                pm25_all_days = []

                for t in range(len(all_aconc_files)):  # Iterate over all time steps (4 days)
                    # Load the ACONC file for each day
                    ds_aconc = xr.open_dataset(all_aconc_files[t])

                    # Compute PM2.5 using ACONC and APMDIAG data
                    ds_apmdiag = ds_apmdiag_list[t]  # Select the corresponding APMDIAG dataset
                    pm25_day = compute_pm25_from_aconc_apmdiag_for_all_days(ds_aconc, ds_apmdiag)

                    pm25_all_days.append(pm25_day)

                # Stack the PM2.5 data for all days (96 hours total)
                pm25_all_days = np.stack(pm25_all_days, axis=0)
                print(pm25_all_days.shape)

                # Get the number of files (days)
                num_files = len(file_list)
                num_time_steps = 24 * num_files  # Total time steps (24 hours per day * number of files)

                # Extract the necessary dimensions (nz, nx, ny)
                nz, nx, ny = pm25_all_days.shape[2], pm25_all_days.shape[3], pm25_all_days.shape[4]

                # Reshape the pm25_all_days array to match the new time steps
                pm25_reshaped = pm25_all_days.reshape(num_time_steps, nz, nx, ny)

                # Create a new TSTEP coordinate for the reshaped PM2.5 data (it should have 96 time steps)
                tstep_values = np.arange(num_time_steps)  # Create an array of length num_time_steps (96 for 4 days)

                # Create a new DataArray for PM2.5 with the reshaped data
                pm25_data_array = xr.DataArray(
                    pm25_reshaped,
                    dims=("TSTEP", "LAY", "ROW", "COL"),
                    coords={
                        "TSTEP": tstep_values,  # Use the new TSTEP coordinate
                        "LAY": ds_aconc["LAY"].values,  # Assuming 'LAY' exists in your ACONC dataset
                        "ROW": ds_aconc["ROW"].values,  # Assuming 'ROW' exists in your ACONC dataset
                        "COL": ds_aconc["COL"].values,  # Assuming 'COL' exists in your ACONC dataset
                    },
                    attrs={"long_name": "PM2.5", "units": "µg/m³"}
                )

                # Add the PM2.5 DataArray to the dataset
                ds["PM2.5"] = pm25_data_array

                # Now PM2.5 is part of the dataset, and you should be able to select it without any dimension errors

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
        #lon2d, lat2d = generate_latlon_grid_from_griddesc(griddesc_file, "WRFtestCMAQ")
        #griddesc_file = "/home/duch/cmaq/khaliaforecast/20240207_M200/esme_local/forecast/run/MCIP/2024-02-07/d01/GRIDDESC"
        #lon2d, lat2d, grid_info = generate_latlon_grid_from_griddesc(griddesc_file, "WRFtestCMAQ")
        # find the GRIDDESC for chosen domain & date (example uses first date found)
        griddesc_file = os.path.join(first_date, domain, "GRIDDESC")
        lon2d, lat2d, grid_info = generate_latlon_grid_from_griddesc(griddesc_file)

        #st.sidebar.write(f"Grid resolution: {grid_info['xcell']/1000:.1f} km")
        #st.sidebar.write(f"Grid size: {grid_info['nx']} x {grid_info['ny']}")

        print(f"Grid resolution: {grid_info['xcell']/1000:.1f} km")
        print(f"Grid size: {grid_info['nx']} x {grid_info['ny']}")

        # Load WRF files once for animation or static
        wrf_files = sorted(glob.glob(os.path.join(wrf_base, f"met_em.{domain}.*.nc")))
        #wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time", engine="netcdf4")
        #wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time", engine="pseudonetcdf")
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
                    #fig = plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind)
                    # Plot concentration with wind
                    fig = plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind=True, nx=grid_info["nx"], ny=grid_info["ny"])

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
            # fig = plot_concentration_with_wind(ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind)
            fig = plot_concentration_with_wind(
                  ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind=True, nx=grid_info["nx"], ny=grid_info["ny"]
            )
            st.pyplot(fig)

        st.sidebar.write(f"Selected time: {times[time_index]}")

        # --- Folium map for interaction ---
        st.subheader("Click on the map to see time series")
        center_lat = np.mean(lat2d)
        center_lon = np.mean(lon2d)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5)

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
            "PM2.5": "PM2.5",
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
            if parameter_code.lower() in ("pm2.5", "pm25", "pm_2_5", "pm-2.5"):
                df_merged = pd.merge(df_pred, df_obs, left_index=True, right_index=True, how='inner')
            else:
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

