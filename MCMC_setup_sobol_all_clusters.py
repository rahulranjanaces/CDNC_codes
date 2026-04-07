"""
Monte Carlo Forward Uncertainty Propagation — All 6 Clusters
=============================================================
- Loops through clusters 1–6 sequentially
- Parameters varied: mode1_d, k1, mode2_n, w, mode1_n, k2
- Clusters 4, 5, 6: updraft distribution taken from cluster 1
- All parameter stats computed from cluster-filtered observations
- Saves CSV results and diagnostic figures per cluster
"""

import os
import csv
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from datetime import timedelta
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol
import multiprocessing as mp
from multiprocessing import Pool, Manager

import modules4.execute

# ─────────────────────────────────────────────────────────────────────────────
# 0.  USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
#CLUSTERS_TO_RUN = [1, 2, 3, 4, 5, 6]
CLUSTERS_TO_RUN = [1, 2, 3, 4, 5, 6]
#CLUSTERS_TO_RUN = [ 2, 3]
N_SAMPLES       = 512        # power of 2 — 512 → 5120 CPM runs per cluster
N_CORES         = 8
RESULTS_DIR     = 'sobol_results_all_updated_range_CDNC_mode2_d_wide_updraft_range'
MODEL_PATH      = '/share/rahul_hyytiala/CDNC_project/CDNC/PARSEC-UFO-stockholm_kappa_Köhler5'
UPDRAFT_PATH    = '/share/rahul_hyytiala/CDNC_project/updraft_PDFs/CCN_cycle_harmonized/positive_pdf'

# Parameters to vary in MCMC — mode1_n and k2 added
PARAMS_TO_VARY  = ['mode1_d', 'k1', 'mode2_n', 'w', 'mode1_n', 'k2', 'mode2_d']

# Distribution type per parameter
PARAM_DIST = {
    'mode1_d'    : 'normal',
    'mode1_n'    : 'lognormal',
    'mode2_d'    : 'normal',
    'mode2_n'    : 'lognormal',
    'mode1_sigma': 'normal',
    'mode2_sigma': 'normal',
    'k1'         : 'normal',
    'k2'         : 'normal',
    'w'          : 'lognormal',
}

# Hard physical bounds — Q5/Q95 clipped to these
PARAM_BOUNDS = {
    'mode1_d'  : (10.0,   500.0),
    'mode1_n'  : (10.0,  50000.0),
    'mode2_d'  : (50.0,   800.0),
    'mode2_n'  : (1.0,   10000.0),
    'k1'       : (0.01,    0.90),
    'k2'       : (0.01,    0.90),
    'w'        : (0.01,    6.00),
}

# Cluster thermodynamic settings
CLUSTER_CFG = {
    1: {'init_temp': 285.30, 'init_RH': 92.07, 'init_pres': 97605.29},
    2: {'init_temp': 287.29, 'init_RH': 81.16, 'init_pres': 97773.86},
    3: {'init_temp': 284.61, 'init_RH': 80.86, 'init_pres': 97126.13},
    4: {'init_temp': 283.02, 'init_RH': 79.87, 'init_pres': 98562.87},
    5: {'init_temp': 279.04, 'init_RH': 55.84, 'init_pres': 98045.81},
    6: {'init_temp': 288.89, 'init_RH': 57.57, 'init_pres': 97500.68},
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD FULL (UNFILTERED) DATASETS ONCE
# ─────────────────────────────────────────────────────────────────────────────
def set_index(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.set_index('datetime')

nsd_full  = set_index(pd.read_csv('NSD/NSD_params/NSD_param_scaled.csv'))
comp_raw  = pd.read_csv('comp/comp_opt_kappa_MCMC.csv')
comp_raw.columns = ['datetime', 'k1', 'k2']
comp_full = set_index(comp_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD UPDRAFT DICT ONCE (all PDF files)
# ─────────────────────────────────────────────────────────────────────────────
print("Building updraft file dictionary...")
updraft_files = glob.glob(os.path.join(UPDRAFT_PATH, '*_updraft_pdf.txt'))
updraft_dict  = {}
for f in updraft_files:
    fname  = os.path.basename(f)
    dt_str = fname.split('_updraft_pdf.txt')[0]
    dt     = pd.to_datetime(dt_str, format='%Y%m%d_%H%M%S')
    updraft_dict[dt] = f
print(f"  Found {len(updraft_dict)} updraft PDF files")

def collect_all_updraft(cluster_times):
    vals = []
    for t in cluster_times:
        for file_time, file_path in updraft_dict.items():
            if abs(file_time - t) <= timedelta(hours=2):
                df_up = pd.read_csv(file_path, sep='\t')
                df_up.columns = ['w', 'pdf']

                sampled = np.repeat(
                    df_up['w'].values,
                    (df_up['pdf'] * 1000).astype(int)
                )
                vals.extend(sampled)

    vals = np.array(vals)
    return vals[vals > 0]  # keep only positive updrafts


# ─────────────────────────────────────────────────────────────
# COLLECT UPDRAFT from ALL CLUSTERS
# ─────────────────────────────────────────────────────────────
cluster_files = sorted(glob.glob('NSD/NSD_obs/cluster*.csv'))

all_updrafts_dict = {}
all_updrafts_list = []

for file in cluster_files:
    cluster_name = os.path.basename(file).replace('.csv', '')
    
    df = pd.read_csv(file)
    df = set_index(df)  # your existing function
    
    updraft_vals = collect_all_updraft(df.index)

    all_updrafts_dict[cluster_name] = updraft_vals
    all_updrafts_list.extend(updraft_vals)

    if len(updraft_vals) == 0:
        print(f"{cluster_name}: N=0 (no valid updraft data)")
    else:
        print(f"{cluster_name}: N={len(updraft_vals)}, "
              f"mean={np.mean(updraft_vals):.4f}, "
              f"median={np.median(updraft_vals):.4f}")

# Convert to single array if needed
all_updrafts_array = np.array(all_updrafts_list)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER — collect updraft samples for a given cluster's timestamps
# ─────────────────────────────────────────────────────────────────────────────
def collect_updraft(cluster_times):
    vals = []
    for t in cluster_times:
        for file_time, file_path in updraft_dict.items():
            if abs(file_time - t) <= timedelta(hours=2):
                df_up         = pd.read_csv(file_path, sep='\t')
                df_up.columns = ['w', 'pdf']
                sampled       = np.repeat(
                    df_up['w'].values,
                    (df_up['pdf'] * 1000).astype(int)
                )
                vals.extend(sampled)
    vals = np.array(vals)
    return vals[vals > 0]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRE-COMPUTE CLUSTER 1 UPDRAFT — used for clusters 4, 5, 6
# ─────────────────────────────────────────────────────────────────────────────
print("\nPre-computing cluster 1 updraft distribution (for clusters 4–6)...")
cls1_df           = set_index(pd.read_csv('NSD/NSD_obs/cluster1.csv'))
#updraft_cls1      = collect_updraft(cls1_df.index)
updraft_cls1      = all_updrafts_array
print(f"  Cluster 1 updraft: N={len(updraft_cls1)}, "
      f"mean={updraft_cls1.mean():.4f}, median={np.median(updraft_cls1):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STAT COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(data, name):
    """Compute (mean, std, median, q5, q95) from a data array."""
    dist   = PARAM_DIST.get(name, 'normal')
    lo, hi = PARAM_BOUNDS.get(name, (-np.inf, np.inf))

    if dist == 'normal':
        mu    = float(np.mean(data))
        sigma = float(np.std(data))
        q5    = mu - 1.645 * sigma
        q95   = mu + 1.645 * sigma

    elif dist == 'lognormal':
        log_d  = np.log(data[data > 0])
        log_mu = float(np.mean(log_d))
        log_sg = float(np.std(log_d))
        mu     = float(np.exp(log_mu))
        sigma  = log_sg
        q5     = float(np.exp(log_mu - 1.645 * log_sg))
        q95    = float(np.exp(log_mu + 1.645 * log_sg))

    q5  = float(max(lo,  q5))
    q95 = float(min(hi,  q95))

    return {'mean': mu, 'std': sigma,
            'median': float(np.median(data)),
            'q5': q5, 'q95': q95, 'dist': dist}

#once, outside the loop
W_STATS_CLUSTER1 = compute_stats(all_updrafts_array, 'w')
print(f"  Cluster 1 w bounds: Q5={W_STATS_CLUSTER1['q5']:.4f}, "
      f"Q95={W_STATS_CLUSTER1['q95']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CPM HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_extra(cfg):
    return {
        'output_type'   : 3,
        'skip_plotting' : True,
        'initial_height': 150,
        'nmodes'        : 1,
        'rebin_type'    : 3,
        'n_bins'        : 400,
        'logidx'        : [],
        'cloud_depth'   : 1800,
        'surf_tension'  : 72.8,
        'model_path'    : MODEL_PATH,
        'init_temp'     : cfg['init_temp'],
        'init_RH'       : cfg['init_RH'],
        'init_pres'     : cfg['init_pres'],
    }

def run_cpm_single(param_row, param_names, FIXED_PARAMS, cfg, workdir):
    sampled = {name: param_row[i] for i, name in enumerate(param_names)}
    p       = {**FIXED_PARAMS, **sampled}

    Extra            = build_extra(cfg)
    Extra['updraft'] = p['w']
    Extra['inputs']  = np.array([
        p['mode1_n']     * 1e6,
        p['mode1_d'] / 2 * 1e-3,
        p['mode1_sigma'],
        p['mode2_n']     * 1e6,
        p['mode2_d'] / 2 * 1e-3,
        p['mode2_sigma'],
        cfg['init_RH'],
        72.8,
        p['k1'],
        p['k2'],
        p['w'],
    ], dtype=float)

    try:
        raw = modules4.execute.execute(Extra, workdir=workdir)
        return float(raw['fa_act_max'])
    except Exception as e:
        print(f"  CPM failed: {e}")
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PARALLEL WORKER
# ─────────────────────────────────────────────────────────────────────────────
def worker_chunk(chunk_indices, param_values, param_names,
                 FIXED_PARAMS, cfg, output_file, lock):
    exec_name = "parsec-ufo"
    with tempfile.TemporaryDirectory(prefix="parsec_mc_") as workdir:
        shutil.copy(os.path.join(MODEL_PATH, exec_name), workdir)
        shutil.copytree(
            os.path.join(MODEL_PATH, "inputs"),
            os.path.join(workdir, "inputs")
        )
        for idx in chunk_indices:
            cdnc = run_cpm_single(
                param_values[idx], param_names, FIXED_PARAMS, cfg, workdir
            )
            with lock:
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([idx] + list(param_values[idx]) + [cdnc])

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN CLUSTER LOOP
# ─────────────────────────────────────────────────────────────────────────────
mp.set_start_method('fork', force=True)

if __name__ == '__main__':

    for CLUSTER_NUMB in CLUSTERS_TO_RUN:

        print(f"\n{'='*65}")
        print(f"  CLUSTER {CLUSTER_NUMB}")
        print(f"{'='*65}")

        cfg = CLUSTER_CFG[CLUSTER_NUMB]

        # cluster-specific output subdirectory
        cls_dir = os.path.join(RESULTS_DIR, f'cluster_{CLUSTER_NUMB}')
        os.makedirs(cls_dir, exist_ok=True)

        # ── load cluster timestamps ───────────────────────────────────────────
        cls_df = set_index(
            pd.read_csv(f'NSD/NSD_obs/cluster{CLUSTER_NUMB}.csv')
        )

        # ── filter NSD and comp to cluster timestamps ─────────────────────────
        nsd_c = pd.concat([nsd_full, cls_df], axis = 1).dropna().iloc[:, :6]
        comp_c = pd.concat([comp_full, cls_df], axis = 1).dropna().iloc[:, :2]

        print(f"  NSD obs  : {len(nsd_c)} rows")
        print(f"  Comp obs : {len(comp_c)} rows")

        # ── updraft: clusters 1–3 use own, clusters 4–6 use cluster 1 ─────────
        if CLUSTER_NUMB <= 0:
        #if CLUSTER_NUMB <= 0:
            updraft_values = collect_updraft(cls_df.index)
            print(f"  Updraft  : own cluster ({len(updraft_values)} samples)")
        else:
            updraft_values = all_updrafts_array.copy() # updraft_cls1.copy()
            print(f"  Updraft  : from cluster 1 ({len(updraft_values)} samples)")

        print(f"  w mean={updraft_values.mean():.4f}  "
              f"median={np.median(updraft_values):.4f}  "
              f"std={updraft_values.std():.4f}")

        # ── compute stats for all parameters ─────────────────────────────────
        def get_data(name):
            if name in nsd_c.columns:
                return nsd_c[name].dropna().values
            elif name in comp_c.columns:
                return comp_c[name].dropna().values
            elif name == 'w':
                return updraft_values
            else:
                raise ValueError(f"Unknown param: {name}")
        
        
        all_param_names = list(nsd_c.columns) + list(comp_c.columns) + ['w']
        ALL_STATS = {}
        for name in all_param_names:
            if name == 'w' and CLUSTER_NUMB > 0:
                # reuse cluster 1 updraft bounds
                ALL_STATS[name] = W_STATS_CLUSTER1
                continue
            try:
                ALL_STATS[name] = compute_stats(get_data(name), name)

            except Exception as e:
                    print(f"  Warning: {name}: {e}")

        print(f"\n  Parameter statistics (cluster {CLUSTER_NUMB}):")
        print(f"  {'param':15s}  {'mean':>10s}  {'std':>10s}  "
              f"{'Q5':>10s}  {'Q95':>10s}")
        for name, s in ALL_STATS.items():
            print(f"  {name:15s}  {s['mean']:10.4g}  {s['std']:10.4g}  "
                  f"{s['q5']:10.4g}  {s['q95']:10.4g}")

        # ── build FIXED and VARIABLE params ──────────────────────────────────
        FIXED_PARAMS = {
            name: ALL_STATS[name]['median']
            for name in all_param_names
            if name not in PARAMS_TO_VARY and name in ALL_STATS
        }
        FIXED_PARAMS['mode1_sigma'] = 1.75
        FIXED_PARAMS['mode2_sigma'] = 1.75

        VARIABLE_PARAMS = {
            name: ALL_STATS[name]
            for name in PARAMS_TO_VARY
        }

        print(f"\n  Fixed params (cluster {CLUSTER_NUMB} medians):")
        for k, v in FIXED_PARAMS.items():
            print(f"    {k:15s} = {v:.4g}")

        print(f"\n  Variable params Q5–Q95:")
        for name, s in VARIABLE_PARAMS.items():
            print(f"    {name:12s}  Q5={s['q5']:.4g}  Q95={s['q95']:.4g}")

        # ── SALib problem ─────────────────────────────────────────────────────
        param_names = list(VARIABLE_PARAMS.keys())
        problem = {
            'num_vars': len(param_names),
            'names'   : param_names,
            'bounds'  : [[VARIABLE_PARAMS[n]['q5'], VARIABLE_PARAMS[n]['q95']]
                         for n in param_names],
        }

        # ── Saltelli samples ──────────────────────────────────────────────────
        param_values = sobol_sampler.sample(
            problem, N_SAMPLES, calc_second_order=False
        )
        n_runs = param_values.shape[0]
        print(f"\n  Total CPM runs: {n_runs}")

        # ── output file ───────────────────────────────────────────────────────
        output_file = os.path.join(
            cls_dir, f'cluster{CLUSTER_NUMB}_mc_samples.csv'
        )
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx'] + param_names + ['fa_act_max'])

        print(f"  Running {n_runs} CPM evaluations on {N_CORES} cores...")

        # ── parallel run ──────────────────────────────────────────────────────
        indices = list(range(n_runs))
        chunks  = np.array_split(indices, N_CORES)

        with Manager() as manager:
            lock = manager.Lock()
            args = [
                (chunk, param_values, param_names,
                 FIXED_PARAMS, cfg, output_file, lock)
                for chunk in chunks
            ]
            with Pool(processes=N_CORES) as pool:
                pool.starmap(worker_chunk, args)

        print(f"  Cluster {CLUSTER_NUMB} CPM runs complete.")

        # ── Sobol analysis ────────────────────────────────────────────────────
        results_df = (pd.read_csv(output_file)
                        .sort_values('idx')
                        .reset_index(drop=True))

        n_failed = results_df['fa_act_max'].isna().sum()
        if n_failed > 0:
            print(f"  Warning: {n_failed} failed runs — replacing with median")
            results_df['fa_act_max'].fillna(
                results_df['fa_act_max'].median(), inplace=True
            )

        Y = results_df['fa_act_max'].values
        valid = np.isfinite(Y)
        Y = Y[valid]
        #X = X[valid]

        print(f"\n  CDNC statistics (cluster {CLUSTER_NUMB}):")
        print(f"    N       : {len(Y)}")
        print(f"    Mean    : {np.mean(Y):.1f} cm⁻³")
        print(f"    Median  : {np.median(Y):.1f} cm⁻³")
        print(f"    Std     : {np.std(Y):.1f} cm⁻³")
        print(f"    5–95pct : {np.percentile(Y,5):.1f} – "
              f"{np.percentile(Y,95):.1f} cm⁻³")

        Si = sobol.analyze(problem, Y, calc_second_order=False,
                           print_to_console=True)

        sobol_df = pd.DataFrame({
            'parameter': param_names,
            'S1'       : Si['S1'],
            'S1_conf'  : Si['S1_conf'],
            'ST'       : Si['ST'],
            'ST_conf'  : Si['ST_conf'],
        })
        sobol_df.to_csv(
            os.path.join(cls_dir, f'cluster{CLUSTER_NUMB}_sobol.csv'),
            index = False
        )
        print(f"\n  Sobol indices (cluster {CLUSTER_NUMB}):")
        print(sobol_df.to_string(index=False))

        # ── figures (unchanged from original) ─────────────────────────────────
        plt.rcParams.update({
            "font.family"     : "sans-serif",
            "font.weight"     : "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold"
        })

        n_p  = len(param_names)
        fig  = plt.figure(figsize=(20, 12), dpi=150)
        gs   = gridspec.GridSpec(2, n_p, hspace=0.45, wspace=0.35,
                                 left=0.06, right=0.97,
                                 top=0.91, bottom=0.09)
        SPINE_LW = 1.8

        # CDNC posterior
        ax_post = fig.add_subplot(gs[0, :n_p-1])
        ax_post.hist(Y, bins=60, color='steelblue', alpha=0.7,
                     density=True, label='MC samples')
        kde_fn = gaussian_kde(Y[~np.isnan(Y)])
        xk     = np.linspace(Y.min(), Y.max(), 300)
        ax_post.plot(xk, kde_fn(xk), 'k-', lw=2.5, label='KDE')
        for pct, ls, col in [
            (5, '--', 'tomato'), (50, '-', 'gold'), (95, ':', 'tomato')
        ]:
            ax_post.axvline(
                np.percentile(Y, pct), color=col, lw=2, ls=ls,
                label=f'p{pct}={np.percentile(Y,pct):.0f} cm⁻³'
            )
        ax_post.set_xlabel('CDNC [cm⁻³]', fontsize=12, fontweight='bold')
        #ax_post.set_xlabel('SS [%]', fontsize=12, fontweight='bold')
        ax_post.set_ylabel('Probability density', fontsize=12,
                           fontweight='bold')
        ax_post.set_title(
            f'CDNC posterior — Cluster {CLUSTER_NUMB}  '
            f'(N={len(Y)}, params: {", ".join(param_names)})',
            fontsize=12, fontweight='bold')
        ax_post.legend(frameon=False,
                       prop={'family':'sans-serif','weight':'bold','size':9})

        # Sobol S1 + ST
        ax_sob = fig.add_subplot(gs[0, n_p-1])
        y_pos  = np.arange(n_p)
        ax_sob.barh(y_pos-0.2, sobol_df['S1'], height=0.35,
                    xerr=sobol_df['S1_conf'], label='S1',
                    color='steelblue', alpha=0.85,
                    error_kw=dict(ecolor='k', capsize=3))
        ax_sob.barh(y_pos+0.2, sobol_df['ST'], height=0.35,
                    xerr=sobol_df['ST_conf'], label='ST',
                    color='tomato', alpha=0.85,
                    error_kw=dict(ecolor='k', capsize=3))
        ax_sob.set_yticks(y_pos)
        ax_sob.set_yticklabels(param_names, fontsize=10, fontweight='bold')
        ax_sob.set_xlabel('Sobol index', fontsize=11, fontweight='bold')
        ax_sob.set_title('S1 vs ST', fontsize=11, fontweight='bold')
        ax_sob.axvline(0, color='k', lw=0.8)
        ax_sob.set_xlim(-0.05, 1.05)
        ax_sob.legend(frameon=False,
                      prop={'family':'sans-serif','weight':'bold','size':9})

        # scatter per param
        for pi, name in enumerate(param_names):
            ax = fig.add_subplot(gs[1, pi])
            ax.scatter(results_df[name], results_df['fa_act_max'],
                       c=results_df['fa_act_max'], cmap='bone',
                       s=3, alpha=0.4, rasterized=True)
            s = VARIABLE_PARAMS[name]
            ax.axvline(s['q5'],  color='tomato', lw=1.2,
                       ls='--', alpha=0.7, label='Q5/Q95')
            ax.axvline(s['q95'], color='tomato', lw=1.2,
                       ls=':', alpha=0.7)
            st_val = sobol_df.loc[
                sobol_df.parameter == name, 'ST'
            ].values[0]
            ax.set_xlabel(name, fontsize=11, fontweight='bold')
            ax.set_ylabel('CDNC', fontsize=11, fontweight='bold')
            ax.set_title(f'ST = {st_val:.2f}', fontsize=11,
                         fontweight='bold')
            ax.legend(frameon=False,
                      prop={'family':'sans-serif','weight':'bold','size':8})

        for ax in fig.get_axes():
            ax.tick_params(labelsize=9, width=SPINE_LW)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
                lbl.set_fontfamily('sans-serif')
            for spine in ax.spines.values():
                spine.set_linewidth(SPINE_LW)

        fig.suptitle(
            f'Monte Carlo Uncertainty Propagation — Cluster {CLUSTER_NUMB}'
            + (' [updraft from cluster 1]'
               if CLUSTER_NUMB > 3 else ''),
            fontsize=15, fontweight='bold')

        fig_path = os.path.join(
            cls_dir, f'cluster{CLUSTER_NUMB}_mc_figures.png'
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Figures saved: {fig_path}")
        print(f"  Cluster {CLUSTER_NUMB} complete.\n")

    print("\nAll clusters complete.")
    print(f"Results in: {RESULTS_DIR}/cluster_{{1..6}}/")