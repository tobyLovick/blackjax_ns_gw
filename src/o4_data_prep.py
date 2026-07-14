"""
Download and preprocess strain data + PSDs for one O4 event.

Per-event output in  o4_data/{event_name}/:
  frequencies.npy          analysis frequency grid (20-1024 Hz, df=0.25 Hz)
  {DET}_strain.npy         complex FFT of on-source 4s segment
  {DET}_psd_welch.npy      Welch median PSD on analysis grid
  {DET}_psd_bw.npy         BayesWave median PSD interpolated to analysis grid

Run on a login node (needs internet). Can be arrayed:
  python o4_data_prep.py --event-idx $SLURM_ARRAY_TASK_ID --event-list o4_event_list.json
"""

import os, sys, json, argparse, shutil, time
import numpy as np
import urllib.request
import tempfile
import scipy.signal


def _mb(path):
    return os.path.getsize(path) / 1e6


def _t(label, t0):
    print(f"    {label}: {time.time()-t0:.1f}s", flush=True)

SAMPLE_RATE       = 4096    # Hz
DURATION          = 4       # s — on-source segment
POST_TRIGGER      = 2       # s after GPS trigger
F_LOW             = 20.0
F_HIGH            = 512.0
DOWNLOAD_DURATION = 64      # s of strain to download for Welch PSD

DETECTORS = ['H1', 'L1', 'V1']   # K1 not in jimgw 0.2.0; require ≥2 of these


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--event-idx',  type=int, default=None)
    p.add_argument('--event-name', type=str, default=None)
    p.add_argument('--event-list', type=str, default='o4_event_list.json')
    return p.parse_args()


def load_event(args):
    with open(args.event_list) as f:
        events = json.load(f)
    if args.event_idx is not None:
        return events[args.event_idx]
    for ev in events:
        if ev['name'] == args.event_name:
            return ev
    raise ValueError(f"Event {args.event_name} not found in {args.event_list}")


def make_analysis_freqs():
    """Frequency array for the 4s on-source FFT, filtered to [F_LOW, F_HIGH]."""
    n_samples = DURATION * SAMPLE_RATE
    freqs_full = np.fft.rfftfreq(n_samples, d=1.0/SAMPLE_RATE)
    mask = (freqs_full >= F_LOW) & (freqs_full <= F_HIGH)
    return freqs_full[mask], mask


def download_strain(det_name, gps_start, gps_end):
    from gwpy.timeseries import TimeSeries
    print(f"  [{det_name}] Downloading {DOWNLOAD_DURATION}s strain ...", flush=True)
    t0 = time.time()
    try:
        ts = TimeSeries.fetch_open_data(det_name, gps_start, gps_end,
                                        sample_rate=SAMPLE_RATE, cache=False)
        nbytes = ts.value.nbytes
        _t(f"done  ({nbytes/1e6:.1f} MB in memory)", t0)
        return ts
    except Exception as e:
        print(f"  WARNING: could not download {det_name}: {e}")
        return None


def compute_welch_psd(ts_long, gps_trigger, freqs_analysis):
    """
    Welch median PSD from off-source portions of a long strain segment.
    Excludes ±POST_TRIGGER s around the trigger.
    """
    import gwpy.frequencyseries
    sample_rate = int(ts_long.sample_rate.value)
    dt = ts_long.t0.value

    # Off-source mask: exclude [trigger - POST_TRIGGER, trigger + POST_TRIGGER]
    t = np.arange(len(ts_long)) / sample_rate + dt
    on_start = gps_trigger - POST_TRIGGER
    on_end   = gps_trigger + POST_TRIGGER
    off_mask = (t < on_start) | (t > on_end)
    data_off = ts_long.value[off_mask]

    nperseg = DURATION * sample_rate   # df = 1/DURATION = 0.25 Hz
    noverlap = nperseg // 2
    f_welch, psd = scipy.signal.welch(data_off, fs=sample_rate,
                                       nperseg=nperseg, noverlap=noverlap,
                                       window='hann', average='median')
    # Interpolate to analysis grid
    psd_interp = np.interp(freqs_analysis, f_welch, psd)
    return psd_interp


def extract_on_source_fft(ts_long, gps_trigger, freqs_mask):
    """Crop 4s on-source segment and FFT."""
    from gwpy.timeseries import TimeSeries
    t_start = gps_trigger - POST_TRIGGER
    t_end   = t_start + DURATION
    ts_on = ts_long.crop(t_start, t_end)

    # Tukey window
    win = scipy.signal.windows.tukey(len(ts_on), alpha=0.2)
    data_win = ts_on.value * win

    fft_full = np.fft.rfft(data_win) / SAMPLE_RATE
    return fft_full[freqs_mask]


def download_bw_psd(hdf5_url, event_name, detectors, freqs_analysis):
    """Download HDF5 from Zenodo, extract BayesWave PSDs, delete HDF5."""
    import h5py

    tmpfile = tempfile.mktemp(suffix='.hdf5', prefix=f'gwtc5_{event_name}_')
    print(f"  Downloading BayesWave HDF5 ...", flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(hdf5_url, tmpfile)
    hdf5_mb = _mb(tmpfile)
    _t(f"done  ({hdf5_mb:.1f} MB)", t0)

    bw_psds = {}
    try:
        with h5py.File(tmpfile, 'r') as f:
            labels = [k for k in f.keys() if k not in ('history', 'version')]
            label  = labels[0]
            avail_dets = list(f[f'{label}/psds'].keys())
            print(f"  HDF5 label: {label}  |  PSDs available: {avail_dets}", flush=True)
            for det in detectors:
                if det not in avail_dets:
                    print(f"  WARNING: {det} not in BW PSD (have {avail_dets})")
                    continue
                raw = f[f'{label}/psds/{det}'][:]
                f_bw   = raw[:, 0]
                psd_bw = raw[:, 1]
                # Strip sentinel/garbage values (last row is often a placeholder)
                valid = (psd_bw > 0) & (psd_bw < 1e-20)
                f_bw, psd_bw = f_bw[valid], psd_bw[valid]
                f_min_bw, f_max_bw = f_bw[0], f_bw[-1]
                print(f"  {det}: BW PSD valid range {f_min_bw:.1f}--{f_max_bw:.1f} Hz  "
                      f"({len(f_bw)} bins, {raw.nbytes/1e3:.1f} kB raw)", flush=True)
                bw_psds[det] = np.interp(freqs_analysis, f_bw, psd_bw,
                                         left=psd_bw[0], right=psd_bw[-1])
    finally:
        os.remove(tmpfile)
        print(f"  HDF5 deleted  (kept {sum(v.nbytes for v in bw_psds.values())/1e3:.1f} kB of PSDs)", flush=True)

    return bw_psds, avail_dets


def main():
    args  = parse_args()
    event = load_event(args)

    name      = event['name']
    gps       = event['gps']
    hdf5_url  = event['hdf5_url']
    out_dir   = os.path.join('o4_data', name)
    os.makedirs(out_dir, exist_ok=True)

    freqs_analysis, freqs_mask = make_analysis_freqs()

    freq_path = os.path.join(out_dir, 'frequencies.npy')
    if not os.path.exists(freq_path):
        np.save(freq_path, freqs_analysis)
        print(f"  Saved frequencies ({len(freqs_analysis)} bins, df={freqs_analysis[1]-freqs_analysis[0]:.4f} Hz)")

    # --- Strain + Welch PSD ---
    gps_start = gps - DOWNLOAD_DURATION / 2
    gps_end   = gps + DOWNLOAD_DURATION / 2
    active_dets = []

    for det in DETECTORS:
        strain_path = os.path.join(out_dir, f'{det}_strain.npy')
        welch_path  = os.path.join(out_dir, f'{det}_psd_welch.npy')
        if os.path.exists(strain_path) and os.path.exists(welch_path):
            print(f"  {det}: strain + Welch already cached")
            active_dets.append(det)
            continue

        ts = download_strain(det, gps_start, gps_end)
        if ts is None:
            continue

        strain_fft = extract_on_source_fft(ts, gps, freqs_mask)
        psd_welch  = compute_welch_psd(ts, gps, freqs_analysis)

        np.save(strain_path, strain_fft)
        np.save(welch_path, psd_welch)
        active_dets.append(det)
        print(f"  {det}: strain ({_mb(strain_path)*1e3:.1f} kB) + "
              f"Welch PSD ({_mb(welch_path)*1e3:.1f} kB) saved", flush=True)

    if len(active_dets) < 2:
        print(f"  SKIP {name}: only {len(active_dets)} detector(s) available ({active_dets})")
        # Write a flag so the analysis script can skip gracefully
        open(os.path.join(out_dir, 'INSUFFICIENT_DETECTORS'), 'w').close()
        return

    # --- BayesWave PSD ---
    missing_bw = [d for d in active_dets
                  if not os.path.exists(os.path.join(out_dir, f'{d}_psd_bw.npy'))]
    if missing_bw:
        bw_psds, avail_dets = download_bw_psd(hdf5_url, name, active_dets, freqs_analysis)
        for det, psd in bw_psds.items():
            np.save(os.path.join(out_dir, f'{det}_psd_bw.npy'), psd)
            print(f"  {det}: BayesWave PSD saved")
        # Remove any active detector that BW doesn't cover
        active_dets = [d for d in active_dets if d in bw_psds]
    else:
        print(f"  BayesWave PSDs already cached")

    # Record which detectors are actually usable
    with open(os.path.join(out_dir, 'detectors.json'), 'w') as f:
        json.dump(active_dets, f)
    print(f"  {name}: complete. Detectors: {active_dets}")


if __name__ == '__main__':
    main()
