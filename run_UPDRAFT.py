import subprocess
import scipy.stats as stats
from scipy.interpolate import interp1d
import numpy as np
import modules.execute 
import os
import pandas as pd
from datetime import timedelta

def find_chemical_composition(chem_data, target_datetime):
    """
    Finds chemical composition data for a given datetime.
    - If exact match: use it.
    - Else search backwards up to 2 hours in 30 min intervals.
    - If still not found: use default values (org=0.75, inorg=0.20, eBC=0.05).
    """
    chem_data['datetime'] = pd.to_datetime(chem_data['datetime'], format="%Y-%m-%d %H:%M:%S")
    chem_data = chem_data.set_index('datetime').sort_index()

    # Search from exact time backwards in 30 min steps up to 2 hours
    for minutes_back in range(0, 121, 30):
        check_time = target_datetime - timedelta(minutes=minutes_back)
        if check_time in chem_data.index:
            row = chem_data.loc[check_time]
            #print(f"Chemical composition found for {check_time}")
            return row['f_org1'], row['f_AS1'], row['f_BC1'], row['f_org2'], row['f_AS2'], row['f_BC2']

    # If not found, return defaults
    print(f"No chemical composition data found within 2 hours of {target_datetime}. Using defaults.")
    return 0.87, 0.04, 0.09, 0.54, 0.37, 0.09  # Default fractions for mode 1 and mode 2 


def find_modal_params(modal_params_data, target_datetime):
    """
    Finds modal parameters for a given datetime.
    - If exact match: use it.
    - Else search backwards up to 1 hour in 10 min intervals.
    - If still not found: raise an error (since modal params are essential).
    """
    modal_params_data['datetime'] = pd.to_datetime(modal_params_data['datetime'], format="%Y-%m-%d %H:%M:%S")
    modal_params_data = modal_params_data.set_index('datetime').sort_index()

    for minutes_back in range(0, 61, 10):
        check_time = target_datetime - timedelta(minutes=minutes_back)
        if check_time in modal_params_data.index:
            row = modal_params_data.loc[check_time]
            #print(f"Modal parameters found for {check_time}")
            return row

    raise ValueError(f"No modal parameters found within 1 hour of {target_datetime}.")

def find_init_met(met_data, target_datetime):
    """
    Finds meteorological data (RH, P, Temp) for a given datetime.
    - If exact match: use it.
    - Else search backwards up to 60 minutes in 1 min intervals.
    - If still not found: raise an error.
    Assumes columns = [RH, P, Temp] with datetime as index.
    """

    # Search from exact time backwards in 1 min steps up to 30 min
    for minutes_back in range(0, 61, 1):
        check_time = target_datetime - timedelta(minutes=minutes_back)
        if check_time in met_data.index:
            row = met_data.loc[check_time]
            return row['Temp'], row['P'], row['RH']  # Temp in K, P in Pa, RH in %
    
    raise ValueError(f"No meteorological data found within 60 minutes of {target_datetime}.")


def find_init_met_from_profile(met_data_dir, date, target_height=50.0):
    import os, glob, pandas as pd

    # Ensure date is a string in the correct format
    if not isinstance(date, str):
        date_str = date.strftime("%Y%m%d_%H%M")  # e.g. 20140510_0200
    else:
        date_str = date

    # --- find the matching file ---
    pattern = os.path.join(met_data_dir, f"*{date_str}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file found for date {date_str} in {met_data_dir}")
    file = files[0]

    # --- load the file ---
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = [col.lower() for col in df.columns]

    # Required columns
    required_cols = {'height_m', 'temperature_k', 'pressure_pa', 'relative_humidity_%'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {file} does not contain the required columns {required_cols}")

    # --- extract values at target height ---
    if target_height not in df['height_m'].values:
        raise ValueError(f"Target height {target_height} m not found in {file}")

    row = df[df['height_m'] == target_height].iloc[0]
    
    # --- save environmental temperature profile file ---
    save_dir = "/Users/rahulranjan/CDNC_project/CDNC/PARSEC-UFO-stockholm_to_run/inputs"
    save_path = os.path.join(save_dir, "environmental_temperature_profile_user.dat")

    # Select only height and temperature, no header, no index
    df[['height_m', 'temperature_k']].to_csv(
        save_path, sep=" ", index=False, header=False
    )

    return row['temperature_k'], row['pressure_pa'], row['relative_humidity_%']


import numpy as np
from scipy.interpolate import interp1d

def get_profiles_vs_w(Extra, w):
    """
    Run PARSEC-UFO for each updraft velocity in w and return
    interpolated profiles (CDNC, SS, height) on a common height grid.
    """
    Height_profiles = []
    SS_profiles = []
    CDNC_profiles = []

    # run CPM for each w
    for j in range(len(w)):
        Extra_local = Extra.copy()
        Extra_local["updraft"] = w[j]
        raw = modules.execute.execute(Extra_local)

        Height_profiles.append(np.array(raw['height_profile']))
        SS_profiles.append(np.array(raw['SS_amb_profile']))
        # CDNC profiles are the sum of the two modes
        CDNC_profiles.append(raw['fa_act_profile_mode1'] + raw['fa_act_profile_mode2'])

    # ---- build common grid ----
    z_min = min(np.min(h) for h in Height_profiles)
    z_max = max(np.max(h) for h in Height_profiles)
    spacing_per_profile = [np.mean(np.diff(h)) for h in Height_profiles]
    avg_spacing = np.mean(spacing_per_profile)

    num_points = int((z_max - z_min) / avg_spacing) + 1
    z_common = np.linspace(z_min, z_max, num_points)[:-4]

    # ---- interpolate all runs to common grid ----
    nW = len(w)
    cdnc_interp = np.zeros((nW, len(z_common)))
    ss_interp = np.zeros((nW, len(z_common)))

    for i in range(nW):
        f_cdnc = interp1d(Height_profiles[i], CDNC_profiles[i],
                          bounds_error=False, fill_value=0)
        f_ss   = interp1d(Height_profiles[i], SS_profiles[i],
                          bounds_error=False, fill_value=0)
        cdnc_interp[i, :] = f_cdnc(z_common)
        ss_interp[i, :]   = f_ss(z_common)

    return z_common, cdnc_interp, ss_interp

from joblib import Parallel, delayed

def run_single_w(Extra, w_val):
    Extra_local = Extra.copy()
    Extra_local["updraft"] = w_val
    raw = modules.execute.execute(Extra_local)
    return (np.array(raw['height_profile']),
            np.array(raw['SS_amb_profile']),
            raw['fa_act_profile_mode1'] + raw['fa_act_profile_mode2'])

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d


def get_profiles_vs_w_parallel(Extra, w, n_jobs=-1):
    """
    Run PARSEC-UFO for each updraft velocity in w (in parallel).
    Returns:
      - cleaned_profiles: [nW x nZ x 3] array (z, cdnc, ss), NaN-padded
      - z_profiles: raw height profiles from each run
      - ss_profiles: raw supersaturation profiles
      - cdnc_profiles: raw CDNC profiles
    """

    # Run all cases in parallel
    results = Parallel(n_jobs = n_jobs)(
        delayed(run_single_w)(Extra, w_val) for w_val in w
    )
    z_profiles, ss_profiles, cdnc_profiles = zip(*results)

    # Build common z grid
    z_min = min(np.min(h) for h in z_profiles)
    z_max = max(np.max(h) for h in z_profiles)
    spacing_per_profile = [np.mean(np.diff(h)) for h in z_profiles]
    avg_spacing = np.mean(spacing_per_profile)

    num_points = int((z_max - z_min) / avg_spacing) + 1
    z_common = np.linspace(z_min, z_max, num_points)

    nW = len(w)
    #profile_interp = np.zeros((nW, len(z_common), 3))
    cdnc_interp = np.zeros((nW, len(z_common)))
    ss_interp = np.zeros((nW, len(z_common)))

    for i in range(nW):
        #f_cdnc = interp1d(z_profiles[i], cdnc_profiles[i],
        #                  bounds_error = False, fill_value = np.nan)
        f_cdnc = interp1d(z_profiles[i], cdnc_profiles[i],
                          bounds_error = False, fill_value = None)
        #f_ss   = interp1d(z_profiles[i], ss_profiles[i],
        #                  bounds_error = False, fill_value = np.nan)
        f_ss   = interp1d(z_profiles[i], ss_profiles[i],
                          bounds_error = False, fill_value = None)

        cdnc_interp[i, :] = f_cdnc(z_common)
        ss_interp[i, :]   = f_ss(z_common)

        # Drop NaNs across all three columns
        #prof = profile_interp[i, :, :]
        #mask = np.all(np.isfinite(prof), axis = 1)
        #prof_clean = prof[mask, :]

        #cleaned_profiles_list.append(prof_clean)
        #if prof_clean.shape[0] > max_len:
         #   max_len = prof_clean.shape[0]

    # Convert list -> array with NaN padding
    #cleaned_profiles = profile_interp#np.full((nW, max_len, 3), np.nan)
    #for i, prof in enumerate(cleaned_profiles_list):
    #    cleaned_profiles[i, :prof.shape[0], :] = prof

    #return (
    #    cleaned_profiles,  # [nW x nZ x 3] padded with NaN
    #    z_profiles,
    #    ss_profiles,
    #    cdnc_profiles
    #)
    return (
        z_common,
        cdnc_interp, # shape (nW, nZ) 
        ss_interp,   # shape (nW, nZ)
        z_profiles,
        ss_profiles,
        cdnc_profiles
    )

def cdnc_method_mean_w(Extra, w, pdf_w):
    """Eq. 1: Nd at mean positive w (run parcel model directly at mean w)."""
    Extra_local = Extra.copy()
    mean_w = np.trapz(w * pdf_w, w) / np.trapz(pdf_w, w)
    Extra_local["updraft"] = mean_w
    
    raw = modules.execute.execute(Extra_local)
    #idx_max = np.argmax(raw['SS_amb_profile'])
    # Find index of max SS on the profile
    net_CDNC_profile = raw['fa_act_profile_mode1'] + raw['fa_act_profile_mode2']
    return raw['SS_amb_profile'], net_CDNC_profile

def cdnc_method_characteristic_w(Extra, w, pdf_w, lam = 0.6):
    """Eq. 2: Nd at w* = λσ (run parcel model directly at characteristic w)."""
    Extra_local = Extra.copy()

    mean_w = np.trapz(w * pdf_w, w) / np.trapz(pdf_w, w)
    sigma = np.sqrt(np.trapz(((w - mean_w) ** 2) * pdf_w, w) / np.trapz(pdf_w, w))
    w_star = lam * sigma

    Extra_local["updraft"] = w_star
    raw = modules.execute.execute(Extra_local)

    # Find index of max SS on the profile
    net_CDNC_profile = raw['fa_act_profile_mode1'] + raw['fa_act_profile_mode2']
    return raw['SS_amb_profile'], net_CDNC_profile

def cdnc_method_pdf_weighted(Extra, w, pdf_w):
    """
    PDF-weighted integration of CDNC and SS profiles.

    Steps:
      1. Interpolate all runs to a common height grid (cleaned).
      2. For each height z, integrate CDNC(w,z) and SS(w,z) over p(w).
      3. Return PDF-weighted profiles along with interpolated raw profiles.
    """
    # Collect interpolated profiles for each w
    z_common, cdnc_interp, ss_interp, Height_profiles, SS_profiles, CDNC_profiles = get_profiles_vs_w_parallel(Extra.copy(), w)

    # Normalize PDF
    norm = np.trapz(pdf_w, w)
    if norm == 0:
        raise ValueError("PDF normalization is zero — check pdf_w input.")

    # Weighted integration across w (ignoring NaNs)
    
    cdnc_profile_net = np.trapz(cdnc_interp.T * pdf_w, w, axis = 1) / norm
    ss_profile_net   = np.trapz(ss_interp.T   * pdf_w, w, axis = 1) / norm

    return (
        ss_profile_net,   # PDF-weighted SS(z)
        cdnc_profile_net, # PDF-weighted CDNC(z)
        z_common,         # valid height grid
        cdnc_interp,      # interpolated CDNC for each w
        ss_interp,        # interpolated SS for each w
        Height_profiles,  # raw z profiles
        SS_profiles,      # raw SS profiles
        CDNC_profiles     # raw CDNC profiles
    )

def cdnc_method_w_pdf_weighted(w, pdf_w, ss_interp, cdnc_interp):
    """
    Eq. 4: ∫ Nd(w) w p(w) dw / ∫ w p(w) dw
    Steps:
      1. Interpolate all runs to common height grid.
      2. For each height z, integrate CDNC(w,z)*w*p(w) and SS(w,z)*w*p(w).
      3. Divide by ∫ w p(w) dw to normalize.
      4. On the integrated profiles, find height of max SS.
      5. Return CDNC at that height.
    """
    #z_common, cdnc_interp, ss_interp = get_profiles_vs_w_parallel(Extra.copy(), w)

    # Denominator ∫ w p(w) dw
    denom = np.trapz(w * pdf_w, w)
    if denom == 0:
        raise ZeroDivisionError("Denominator ∫ w p(w) dw is zero")

    # Numerators for CDNC and SS (integrated over w for each z)
    cdnc_profile = np.trapz(cdnc_interp.T * (w * pdf_w), w, axis = 1) / denom
    ss_profile   = np.trapz(ss_interp.T   * (w * pdf_w), w, axis = 1) / denom

    # Find index of max SS on the integrated profile
    #idx_max = np.argmax(ss_profile)

    # Return CDNC at that height
    return ss_profile, cdnc_profile


"""what does this code do?
1. It computes the droplet size distribution for a range of updraft velocities
2. It uses the PARSEC-UFO model to simulate the droplet activation
2.1 It reads the updraft velocity and PDF values from a file
2.2 It reads the modal parameters from a file
2.3 It computes the droplet size distribution for each updraft velocity
2.3.1 Definition of cloud droplet:?? finish it before going for runs
2.4 It saves the droplet size distribution to a file
2.5 It repeats the process for different seasons/conditions
3. It saves the droplet size distribution to a file
"""
import csv
import os

output_dir = '/Users/rahulranjan/CDNC_project/CDNC/PDF_method_test/output/'

# Make sure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    # Optional: clear the directory (be cautious!)
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

# Full path to results file
output_file = os.path.join(output_dir, "results.csv")

# Write header if file does not exist
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "max_ss_mean_w", "max_cdnc_mean_w",
            "max_ss_pdf_weighted", "max_cdnc_pdf_weighted",
            "max_ss_w_weighted", "max_cdnc_w_weighted"
        ])


using_server = False
# Define the parameters for the run
Extra = {
"output_type": 1,
"skip_plotting": True,
"cloud_depth": 3000,
"initial_height": 50,
"nmodes": 1,
"rebin_type": 3,  # 3 for ????
"n_bins": 600, # potential impact of 200 vs 400? or going below 200?
"logidx": []
}
#met_file_path = '/Users/rahulranjan/CDNC_project/meteorology/met_all_years.csv'
#met_data = pd.read_csv(met_file_path)
#met_data['datetime'] = pd.to_datetime(met_data['datetime'])
#met_data = met_data.set_index('datetime').sort_index()
#print(met_data)
met_file_path = '/Users/rahulranjan/CDNC_project/temp_profile/2_hourly_interpolated_with_RH'
# read the chemical composition from the file
chem_data = pd.read_csv('/Users/rahulranjan/CDNC_project/comp/optimized/result_nedler_mead_org0p12.csv')
modal_params_data = pd.read_csv('/Users/rahulranjan/CDNC_project/raw_comp_and_NSD_data/NSD_PARAMS_SCALED_unharmonized.CSV')
# LOAD ALL UPDRAFT pdf FILES
updraft_pdf_files_path = '/Users/rahulranjan/CDNC_project/updraft_PDFs/2_hourly'
updraft_pdf_files = [f for f in os.listdir(updraft_pdf_files_path) if f.endswith('.txt')]
# Sort the files to ensure consistent order
updraft_pdf_files.sort()
#print(chem_all_seasons)

n = len(modal_params_data)
max_cdnc_all = []
max_ss_all = []
max_cdnc_weighted_all = []
max_ss_weighted_all = []
max_ss_mean_w_all = []
max_cdnc_mean_w_all = []
case_ids = []  
# Define stop datetime
stop_datetime = pd.to_datetime("2020-12-31 23:50:00")
valid_timesteps = 0   # counter for successful cases
total_timesteps = 0   # counter for all attempted timesteps
# Iterate over updraft PDF files
#for i in range(len(updraft_pdf_files)):
# run few steps for testing
for i in range(1,2):

    print('running case:', i)
    updraft_pdf_file = os.path.join(updraft_pdf_files_path, updraft_pdf_files[i])
    try:
        updraft_pdf = np.loadtxt(updraft_pdf_file, skiprows = 1)
        if updraft_pdf.size == 0:
            print(f"Warning: {updraft_pdf_file} is empty after skipping header. Skipping this case.")
            continue
    except Exception as e:
        print(f"Error reading {updraft_pdf_file}: {e}")
        continue
    
    #print(f"Using updraft PDF file: {updraft_pdf_file}")
    #print(updraft_pdf)

    # Extract just the filename
    filename = os.path.basename(updraft_pdf_file)

    # Remove the suffix and split into date and time
    date_str, time_str, *_ = filename.replace(".txt", "").split("_")

    # Combine into full datetime string
    datetime_str = date_str + time_str  # e.g., "201402050400"

    # Convert to pandas datetime
    date = pd.to_datetime(datetime_str, format="%Y%m%d%H%M")

    # Stop if date exceeds cutoff
    if date > stop_datetime:
        print(f"Reached {date}, stopping iteration.")
        break
    total_timesteps += 1  # count every timestep checked
    #print(f"Extracted datetime string: {datetime_str}")
    #print(f"Extracted datetime object: {date}")

    # Get chemical composition
    org1_frac, inorg1_frac, eBC1_frac, org2_frac, inorg2_frac, eBC2_frac = find_chemical_composition(chem_data, date)

    # Try to get modal parameters, skip iteration if not found
    try:
        modal_params_row = find_modal_params(modal_params_data, date)
    except ValueError as e:
        #print(f"{e} Skipping this case.")
        continue
    
    #print(f"Modal parameters for {date}: {modal_params_row}")
    
    r1 = 1e-3*modal_params_row.iloc[0]/2 # CPM takes radius in micrometer
    #print(f"r1 = {r1}")
    gsd1 = modal_params_row.iloc[1]  # geometric standard deviation of the mode
    n1 = modal_params_row.iloc[2] * 1e6 # number concentration of particles in the mode

    # mode2
    r2 = 1e-3*modal_params_row.iloc[3] / 2
    gsd2 = modal_params_row.iloc[4]
    n2 = modal_params_row.iloc[5] * 1e6
    
    # sequence of chemical species: h2so4, (nh4)hso4, (nh4)2so4, OC, BC, dust, seasalt, SVOC
    Extra['mode1_composition'] = [0.0, 0.0, inorg1_frac, org1_frac, eBC1_frac, 0.0, 0.0]
    Extra['mode2_composition'] = [0.0, 0.0, inorg2_frac, org2_frac, eBC2_frac, 0.0, 0.0]

    #Extra['mode1_composition'] = [0.0, 0.0, chem_all_seasons_ait.iloc[case,1], chem_all_seasons_ait.iloc[case,0], chem_all_seasons_ait.iloc[case,2], 0.0, 0.0]
    #Extra['mode2_composition'] = [0.0, 0.0, chem_all_seasons_acc.iloc[case,1], chem_all_seasons_acc.iloc[case,0], chem_all_seasons_acc.iloc[case,2], 0.0, 0.0]
    #print(Extra['mode1_composition'])
    #print(Extra['mode2_composition'])
    # now read initial temperature, pressure and relative humidity from the file
    # met file
    
    # make a similar function to modal_params_row = find_modal_params(modal_params_data, date) to find the met data where
    # the date is closest to the target date
    try:
        init_temp, init_pres, init_RH = find_init_met_from_profile(met_file_path, date)

        # --- RH condition handling ---
        if 100 <= init_RH <= 101:
            init_RH = 99.0
        elif init_RH > 101 or init_RH < 50:
            print(f"RH = {init_RH:.2f}% > 101 for {date}. Skipping this case.")
            continue
 
    except ValueError as e:
        print(f"{e} Skipping this case.")
        continue

    print(f"Met data found for {date}: Temp = {init_temp:.2f} K, Pressure = {init_pres:.2f} Pa, RH = {init_RH:.2f} %")
    Extra['init_temp'] = init_temp
    Extra['init_pres'] = init_pres
    Extra['init_RH']   = init_RH

    Extra["inputs"] = np.concatenate([
        [n1, r1, gsd1],
        [n2, r2, gsd2],
        [init_RH, 72.8]
    ])
    # If we reached here, all data was available
    valid_timesteps += 1

    # prepare to Run the PARSEC-UFO model
    w = updraft_pdf[:,0] # Get the upDraft velocities for the current season
    pdf_w = updraft_pdf[:,1] # Get the PDF for the current season

    # Filter w and pdf_w before parallel run, the filetr is basically useless currently
    # Filter: keep only pairs where w * pdf_w > 0.02
    mask = (w * pdf_w) > 0.00002
    w_filtered = w[mask]
    pdf_w_filtered = pdf_w[mask]

    # Method 1: Nd at mean positive w
    ss_net_mean_w, CDNC_net_mean_w  = cdnc_method_mean_w(Extra.copy(), w_filtered, pdf_w_filtered)
    idx_max_mean = np.argmax(ss_net_mean_w[:-6])
    #max_ss_mean_w_all.append(ss_net_mean_w[idx_max])
    #max_cdnc_mean_w_all.append(CDNC_net_mean_w[idx_max])
    #print(f"Nd at mean w: {Nd_mean_w}")
    
    # Method 2: Nd at characteristic w* = λσ
    #ss_max_net_charac, CDNC_max_net_charac = cdnc_method_characteristic_w(Extra.copy(), w, pdf_w, lam = 0.6)

    # Method 3: PDF-weighted Nd
    #z_common, ss_max_net_PDF_weighted, CDNC_max_net_PDF_weighted, cdnc_interp, ss_interp  = cdnc_method_pdf_weighted(Extra.copy(), w, pdf_w)
    ss_net, cdnc_net, z_grid, cdnc_interp_all, ss_interp_all, z_raw, ss_raw, cdnc_raw = cdnc_method_pdf_weighted(Extra.copy(), w_filtered, pdf_w_filtered)
    idx_max = np.argmax(ss_net[:-6]) # ignore last 6 points if they are nans
    #max_ss_all.append(ss_net[idx_max])
    #max_cdnc_all.append(cdnc_net[idx_max])
    #case_ids.append(date)

    # Method 4: w·PDF-weighted Nd
    ss_interp_all_w_weighted, cdnc_interp_all_w_weighted = cdnc_method_w_pdf_weighted(w_filtered, pdf_w_filtered, ss_interp_all, cdnc_interp_all)
    idx_max_w_weighted = np.argmax(ss_interp_all_w_weighted[:-6]) # ignore last 6 points if they are nans
    #max_ss_weighted_all.append(ss_interp_all_w_weighted[idx_max_w_weighted])
    #max_cdnc_weighted_all.append(cdnc_interp_all_w_weighted[idx_max_w_weighted])

    # Inside your loop:
    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            date,
            ss_net_mean_w[idx_max_mean], CDNC_net_mean_w[idx_max_mean],
            ss_net[idx_max], cdnc_net[idx_max],
            ss_interp_all_w_weighted[idx_max_w_weighted], cdnc_interp_all_w_weighted[idx_max_w_weighted]
        ])

    # Optional: Nd as a function of updraft velocities
    #Nd_vs_w = get_Nd_vs_w(Extra.copy(), w)

# to know for how many timesteps we have all the data available
print(f"\nTotal timesteps checked: {total_timesteps}")
print(f"Valid timesteps with all data available: {valid_timesteps}")
if total_timesteps > 0:
    fraction = valid_timesteps / total_timesteps
    print(f"Fraction of valid timesteps: {fraction:.2%}")
    #288.551