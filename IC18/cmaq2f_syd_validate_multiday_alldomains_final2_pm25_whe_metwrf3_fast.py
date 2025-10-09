import xarray as xr
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
import folium
import glob
import time
import os
import requests
from datetime import datetime, timedelta

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import fnmatch

# Domain selection
domain = "d02"

# Paths
cmaq_base = "/mnt/scratch_lustre/ar_policy/whe_project/esme_local/jul13_IC18_adj/run/CTM"

mcip_base = "/mnt/scratch_lustre/ar_policy/whe_project/esme_local/jul13_IC18_adj/run/MCIP"

wrf_base = "/mnt/scratch_lustre/ar_policy/whe_project/esme_local/wrf_gmr_2013/run/WPS"

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
        print(f"Station '{station_name}' not found in API.")
        return None
    if parameter_code not in param_map:
        print(f"Parameter '{parameter_code}' not available for hourly data.")
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
        print(f"⚠️ Failed to fetch data from API: {e}")
        return None

    records = [
        {
            "datetime": datetime.strptime(r["Date"], "%Y-%m-%d") + timedelta(hours=r["Hour"]),
            "obs": r["Value"]
        }
        for r in resp if r["Value"] is not None
    ]

    if not records:
        print(f"No valid data returned from API for {station_name}")
        return None

    df_obs = pd.DataFrame(records).set_index("datetime").sort_index()
    return df_obs

from sklearn.metrics import mean_squared_error

def compute_metrics(df):
    obs = df["Observed"]
    pred = df["Model"]

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

def find_apmdiag_for_aconc(aconc_path):
    """Given path to a CCTM_ACONC file or to a directory containing them,
    try to find the corresponding CCT_AELMO* file in the same directory."""
    
    if os.path.isdir(aconc_path):
        base_dir = aconc_path
    else:
        base_dir = os.path.dirname(aconc_path)
    
    # Debugging: Print the directory we are searching
    print(f"Looking for AELMO in: {base_dir}")
    
    # Look for files named like CCT_APMDIAG*.nc in that directory
    for fname in sorted(os.listdir(base_dir)):
        print(f"Checking file: {fname}")  # Debugging: Print file names found in the directory
        if fname.startswith("CCTM_AELMO"):  # Match the prefix directly
            print(f"Found AELMO file: {fname}")  # Debugging: Found the correct file
            return os.path.join(base_dir, fname)
    
    # Debugging: If no file is found
    print("AELMO file not found!")
    return None

def compute_pm25_from_aelmo_for_all_days(ds_apmdiag):
    """Compute PM2.5 for all days using the formula and available data."""
    # Initialize an array to store PM2.5 for all days
    pm25_all_days = []
    pm10_all_days = []

    # Loop through all time steps (TSTEP)
    for t in range(ds_apmdiag.sizes['TSTEP']):
        # Extract relevant variables from ACONC for the current time step
        pm25 = ds_apmdiag["PM25"].isel(TSTEP=t).values
        pm10 = ds_apmdiag["PM10"].isel(TSTEP=t).values
        # Append the PM2.5 values for this time step to the list
        pm25_all_days.append(pm25)
        pm10_all_days.append(pm10)

    # Convert the list to a numpy array and stack it as a dataset
    pm25_all_days = np.stack(pm25_all_days, axis=0)  # Shape (TSTEP, LAY, ROW, COL)
    pm10_all_days = np.stack(pm10_all_days, axis=0)  # Shape (TSTEP, LAY, ROW, COL)

    return pm25_all_days, pm10_all_days

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
    if nx < 185 and ny < 250:
        return 20  # Increase step size (fewer arrows) for smaller domains
    elif nx < 200 and ny < 350:
        return 30  # A moderate step for mid-sized domains
    else:
        return 40  # Default step for larger domains like d01

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
            #u_unstaggered = 0.5 * (uu[:, :-1] + uu[:, 1:])
            #v_unstaggered = 0.5 * (vv[:-1, :] + vv[1:, :])
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
            #step = calculate_step_size(nx, ny)  # Calculate the step based on grid size
            step = max(1, calculate_step_size(nx, ny) // 2)
            ax.quiver(
                lon2d[::step, ::step],
                lat2d[::step, ::step],
                u_trimmed[::step, ::step],
                v_trimmed[::step, ::step],
                transform=ccrs.PlateCarree(),
                scale=100,
                width=0.002,
                color="white",
                alpha=0.7,
                zorder=2
            )

        except Exception as e:
            print(f"Could not overlay wind vectors: {e}")

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


# --- Find GRIDDESC file for chosen domain ---
date_dirs = sorted(glob.glob(os.path.join(mcip_base, "2013-*")))
if not date_dirs:
    print(f"No date directories found in {mcip_base}")
else:
    # Pick first date folder
    first_date = date_dirs[0]
    griddesc_file = os.path.join(first_date, domain, "GRIDDESC")
    if os.path.exists(griddesc_file):
        grid_info = read_griddesc(griddesc_file)
        if grid_info:
            print(f"Grid resolution: {grid_info['xcell']/1000:.1f} km")
            print(f"Grid size: {grid_info['nx']} x {grid_info['ny']}")
        else:
            print(f"Could not parse grid info from {griddesc_file}")
    else:
        print(f"GRIDDESC not found: {griddesc_file}")

# --- File discovery ---
if os.path.exists(cmaq_base):
    file_list = sorted(glob.glob(
        #os.path.join(cmaq_base, f"2013-07-2*/{domain}/CCTM_ACONC_v54_intel_{domain}_*.nc")
        os.path.join(cmaq_base, f"2013-07-*/{domain}/CCTM_ACONC_v54_intel_{domain}_*.nc")
    ))
    print(f"Found {len(file_list)} CMAQ files.")

    wrf_files = sorted(glob.glob(
        #os.path.join(wrf_base, f"met_em.{domain}.2013-07-2[0-5]*nc")
        os.path.join(wrf_base, f"met_em.{domain}.2013-07-*nc")
    ))
    print(f"Found {len(wrf_files)} WRF files.")
else:
    print("Please check your CMAQ base directory path.")

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

        print("CMAQ file loaded.")

        # --- Sidebar for variable selection ---
        variables = [v for v in ds.data_vars if v != "TFLAG"] + ['PM2.5'] + ["PM10"]  # Add PM2.5 as an option
        variable = "PM2.5"

        if variable.lower() in ("pm2.5", "pm25", "pm_2_5", "pm10"):
            # Find all ACONC and AELMO files
            all_aconc_files = file_list if file_list else None
            all_apm_files = []

            for aconc_file in all_aconc_files:
                # Find the corresponding AELMO file for each ACONC file
                apm_file = find_apmdiag_for_aconc(aconc_file)
                if apm_file and os.path.exists(apm_file):
                    all_apm_files.append(apm_file)
                else:
                    printf(f"AELMO file not found for {aconc_file}; cannot compute PM2.5.")

            if not all_apm_files:
                print("No valid AELMO files found for the given ACONC files.")
            else:
                # Debugging: Show the paths of the AELMO files
                print(f"Found AELMO files: {all_apm_files}")
        
                # Load and process AELMO files
                ds_apmdiag_list = []
                for apm_file in all_apm_files:
                    ds_apmdiag_list.append(xr.open_dataset(apm_file))

                # Now calculate PM2.5 for all days from all ACONC and AELMO datasets
                pm25_all_days = []
                pm10_all_days = []

                for t in range(len(all_aconc_files)):  # Iterate over all time steps (4 days)
                    ds_aconc = xr.open_dataset(all_aconc_files[t]).isel(LAY=[0])
                    # Select only the first layer (LAY=0)
                    # Compute PM2.5 using ACONC and AELMO data
                    ds_apmdiag = ds_apmdiag_list[t]  # Select the corresponding AELMO dataset
                    pm25_day, pm10_day = compute_pm25_from_aelmo_for_all_days(ds_apmdiag)

                    pm25_all_days.append(pm25_day)
                    pm10_all_days.append(pm10_day)

                # Stack the PM2.5 data for all days (96 hours total)
                pm25_all_days = np.stack(pm25_all_days, axis=0)
                pm10_all_days = np.stack(pm10_all_days, axis=0)
                print(pm25_all_days.shape)

                # Get the number of files (days)
                num_files = len(file_list)
                num_time_steps = 24 * num_files  # Total time steps (24 hours per day * number of files)

                # Extract the necessary dimensions (nz, nx, ny)
                nz, nx, ny = pm25_all_days.shape[2], pm25_all_days.shape[3], pm25_all_days.shape[4]

                # Reshape the pm25_all_days array to match the new time steps
                pm25_reshaped = pm25_all_days.reshape(num_time_steps, nz, nx, ny)
                pm10_reshaped = pm10_all_days.reshape(num_time_steps, nz, nx, ny)

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

                pm10_data_array = xr.DataArray(
                    pm10_reshaped,
                    dims=("TSTEP", "LAY", "ROW", "COL"),
                    coords={
                        "TSTEP": tstep_values,  # Use the new TSTEP coordinate
                        "LAY": ds_aconc["LAY"].values,  # Assuming 'LAY' exists in your ACONC dataset
                        "ROW": ds_aconc["ROW"].values,  # Assuming 'ROW' exists in your ACONC dataset
                        "COL": ds_aconc["COL"].values,  # Assuming 'COL' exists in your ACONC dataset
                    },
                    attrs={"long_name": "PM10", "units": "µg/m³"}
                )

                # Add the PM2.5 DataArray to the dataset
                ds["PM2.5"] = pm25_data_array
                ds["PM10"] = pm25_data_array

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
        #wrf_files = sorted(glob.glob(os.path.join(wrf_base, f"met_em.{domain}.2013-07-2[0-5]*nc")))
        wrf_files = sorted(glob.glob(os.path.join(wrf_base, f"met_em.{domain}.2013-07-*nc")))
        #wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time", engine="netcdf4")
        #wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time", engine="pseudonetcdf")
        wrf_ds = xr.open_mfdataset(wrf_files, combine='nested', concat_dim="Time")

        show_wind = False
        animate = False
        if show_wind:
            # Assuming `times` is a list or array of time indices
            for time_index, cmaq_time in enumerate(times):
                # Generate the figure for each time step
                fig = plot_concentration_with_wind(
                    ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind=True, nx=grid_info["nx"], ny=grid_info["ny"]
                )
                # Display the plot for each time step
                plt.title(f"Wind Field at Time: {cmaq_time}")
                plt.show()
                # Optionally, you can save the plots as images (if you don't want to display them interactively)
                # fig.savefig(f"wind_field_{cmaq_time}.png")
                print(f"Plot for time: {cmaq_time} displayed.")

        if animate:
            import matplotlib.animation as animation
            # Create the figure and axis
            fig, ax = plt.subplots()
            # Function to update the plot for each time step (frame)
            def update_plot(frame):
                ax.clear()  # Clear the current axes
    
                # Get the current time index and time step
                time_index = frame
                cmaq_time = times[time_index]
    
                # Call the function to plot the concentration with wind at this time step
                plot_concentration_with_wind(
                    ds, variable, lon2d, lat2d, time_index, wrf_ds, cmaq_time, show_wind=True, nx=grid_info["nx"], ny=grid_info["ny"]
                )
    
                # Set title with the current time step
                ax.set_title(f"Wind Field at Time: {cmaq_time}")
    
                # Optionally, you can adjust the axis limits to match the data range
                ax.set_xlim(lon2d.min(), lon2d.max())
                ax.set_ylim(lat2d.min(), lat2d.max())
    
                # Return the updated plot elements
                return ax,

            # Create the animation
            ani = animation.FuncAnimation(
                fig,            # Figure to animate
                update_plot,    # Update function for each frame
                frames=len(times),  # Number of frames (one for each time step)
                interval=500,   # Time between frames (in ms)
                blit=False,     # Set to False if you want to redraw everything each time
            )

            # Display the animation
            plt.show()

        # Validation and metics
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
            {"name": "WAGGA WAGGA", "lat": -35.1, "lon": 147.358},
            {"name": "WOLLONGONG", "lat": -34.424, "lon": 150.893}
        ]

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

        # Check if the lists are populated properly
        print("Stations:", stations)
        print("Parameters:", PARAMETER_API_MAP)

        # --- Validation Section ---
        print("📡 Model vs Observation Validation")

        default_start = min(times)
        default_end = max(times)
        date_range = [default_start, default_end]

        import seaborn as sns

        # Collect results
        results = []
        for station in stations:
            print(f"Processing station: {station['name']}")  # Print station name for verificati
            station_name = station["name"]
            station_lat = station["lat"]
            station_lon = station["lon"]

            for parameter_code, parameter_api_name in PARAMETER_API_MAP.items():
                try:
                    # --- Model prediction selection ---
                    print(f"Validating {parameter_api_name} for station {station_name}...")
                    print("Date range: ", date_range)
                    start_date, end_date = date_range
                    flat_lats = lat2d.flatten()
                    flat_lons = lon2d.flatten()
                    distances = np.sqrt((flat_lats - station_lat)**2 + (flat_lons - station_lon)**2)
                    min_idx = np.argmin(distances)
                    ny, nx = lat2d.shape
                    i, j = np.unravel_index(min_idx, (ny, nx))

                    series = ds[parameter_code][:, 0, i, j].values
                    df_ts = pd.DataFrame({'Time': times, 'pred': series})
                    #df_pred = pd.DataFrame({"Model": series}, index=pd.to_datetime(times))
                    df_pred = pd.DataFrame({"pred": series}, index=pd.to_datetime(times))
                    # Convert model UTC time to Australia/Sydney local time
                    df_pred = df_pred.tz_localize("UTC").tz_convert("Australia/Sydney")
                    df_pred.index.name = "datetime_local"

                    # --- Observations ---
                    df_obs = fetch_observation_data(
                        station_name=station_name,
                        parameter_code=parameter_api_name,
                        start_date_str=default_start.strftime("%Y-%m-%d"),
                        end_date_str=default_end.strftime("%Y-%m-%d")
                    )
                    # Skip if no data returned
                    if df_obs is None or df_obs.empty:
                        continue
                    df_obs.index = pd.to_datetime(df_obs.index).tz_localize("Australia/Sydney")
                    df_obs.index.name = "datetime_local"

                    # Merge and convert observed units from pphm to ppm if needed
                    if parameter_code.lower() in ("pm2.5", "pm25", "pm_2_5", "pm-2.5", "pm10"):
                        df_merged = pd.merge(df_pred, df_obs, left_index=True, right_index=True, how="inner")
                    else:
                        df_merged = pd.merge(df_pred, df_obs / 100, left_index=True, right_index=True, how="inner")

                    print(f"Observed shape: {df_obs.shape}, Predicted shape: {df_pred.shape}")
                    #df_merged.columns = ["Model", "Observed"]

                    if not df_merged.empty:
                        print(f"### 📊 Comparison at **{station_name}**")

                        fig3, ax3 = plt.subplots()
                        ax3.plot(df_merged.index, df_merged["pred"], label="CMAQ Prediction")
                        ax3.plot(df_merged.index, df_merged["obs"], label="Observation", linestyle='--')
                        ax3.set_title(f"{parameter_code} Concentration at {station_name}")
                        ax3.set_ylabel(f"{parameter_code} ({ds[parameter_code].units})")
                        ax3.set_xlabel("Time")
                        ax3.grid(True)
                        ax3.legend()
                        #plt.show()
                        plt.savefig(f"{parameter_code}_{station_name}.png", dpi=300)
                        plt.close()

                        # --- Compute validation metrics ---
                        #st.line_chart(df_merged.rename(columns={"pred": "Model", "obs": "Observed"}))

                        # Assuming df_merged is a pandas DataFrame
                        df_merged = df_merged.rename(columns={"pred": "Model", "obs": "Observed"})

                        # Plot using Matplotlib
                        plt.plot(df_merged.index, df_merged['Model'], label='Model', color='blue')
                        plt.plot(df_merged.index, df_merged['Observed'], label='Observed', color='red')

                        # Add labels and title
                        plt.xlabel('Time')
                        plt.ylabel(f"{parameter_code} ({ds[parameter_code].units})")
                        plt.title(f"Model vs Observed at {station_name}")

                        # Show the legend
                        plt.legend()

                        # Display the plot
                        #plt.show()
                        plt.savefig(f"{parameter_code}_{station_name}_2.png", dpi=300)
                        plt.close()

                        from sklearn.metrics import mean_squared_error
                        rmse = mean_squared_error(df_merged["Observed"], df_merged["Model"], squared=False)
                        print("RMSE", f"{rmse:.3f} ppm")

                        metrics = compute_metrics(df_merged)
                        print("### 📈 Validation Metrics")
                        print(metrics)
                    else:
                        print("No matching time data found between prediction and observation.")
                    # --- Compute metrics ---
                    rmse = np.sqrt(((df_merged["Model"] - df_merged["Observed"])**2).mean())
                    bias = (df_merged["Model"] - df_merged["Observed"]).mean()
                    corr = df_merged["Model"].corr(df_merged["Observed"])

                    results.append({
                        "Station": station_name,
                        "Parameter": parameter_code,
                        "RMSE": rmse,
                        "Bias": bias,
                        "Corr": corr
                    })

                except Exception as e:
                    print(f"Failed for {station_name} ({parameter_code}): {e}")
                    continue

        # Convert to DataFrame
        metrics_df = pd.DataFrame(results)
        # Pivot for heatmap (stations as rows, parameters as columns)
        rmse_matrix = metrics_df.pivot(index="Station", columns="Parameter", values="RMSE")
        bias_matrix = metrics_df.pivot(index="Station", columns="Parameter", values="Bias")
        corr_matrix = metrics_df.pivot(index="Station", columns="Parameter", values="Corr")

        plt.figure(figsize=(10, 6))
        sns.heatmap(rmse_matrix, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("RMSE Heatmap (Model vs Observation)")
        plt.ylabel("Station")
        plt.xlabel("Parameter")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"RMSE_heatmap_{domain}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.heatmap(bias_matrix, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("Bias Heatmap (Model vs Observation)")
        plt.ylabel("Station")
        plt.xlabel("Parameter")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"Bias_heatmap_{domain}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("Corr Heatmap (Model vs Observation)")
        plt.ylabel("Station")
        plt.xlabel("Parameter")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"Corr_heatmap_{domain}.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"Failed to load or plot file: {e}")

