import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
import tkinter as tk
from tkinter import filedialog
import os

# ==============================================================================
# STEP 1: LOAD CSV & EXPLICIT DISCONTINUITY SEGMENTATION (Requirement 2)
# ==============================================================================
def extract_clean_segments(df, min_duration_s=14.0, max_dt_s=0.10):
    """
    Splits the raw recording into contiguous valid segments while handling:
    1. Exact duplicate timestamps (dt == 0.0s).
    2. Isolated 1-2 sample micro-drops in the accelerometer (linearly interpolated).
    3. Missing PPG, 20-bit ADC saturation (>= 1,048,570), or missing Acc blocks.
    4. Un-flagged packet-drop timestamp jumps (dt > 0.10s or dt <= 0.0s).
    """
    # 1. Remove duplicate timestamps (dt == 0)
    dt_raw = df['elapsed_time_s'].diff().fillna(0.025)
    df_clean = df[dt_raw != 0.0].copy().reset_index(drop=True)

    # 2. Interpolate isolated 1-2 sample micro-drops in accelerometer only
    for col in ['acc_x', 'acc_y', 'acc_z']:
        df_clean[col] = df_clean[col].interpolate(method='linear', limit=2)

    # 3. Identify valid rows (PPG present & below 20-bit saturation rail, Acc present)
    is_valid = (
        df_clean['ppg'].notna() &
        (df_clean['ppg'] < 1048570.0) &
        df_clean['acc_x'].notna() &
        df_clean['acc_y'].notna() &
        df_clean['acc_z'].notna()
    )

    # 4. Detect un-flagged timestamp jumps (packet drops without NaNs)
    dt = df_clean['elapsed_time_s'].diff().fillna(0.025)
    time_discontinuity = (dt <= 0.0) | (dt > max_dt_s)

    # 5. Increment segment ID whenever validity flips OR a timestamp jump occurs
    seg_id = ((is_valid != is_valid.shift(fill_value=False)) | time_discontinuity).cumsum()

    valid_rows = df_clean[is_valid]
    segments = []
    for _, grp in valid_rows.groupby(seg_id[is_valid]):
        dur = grp['elapsed_time_s'].iloc[-1] - grp['elapsed_time_s'].iloc[0]
        # Require at least 2s edge trim + one 12s window = 14.0s
        if dur >= min_duration_s:
            segments.append(grp.reset_index(drop=True))

    return segments


# ==============================================================================
# STEP 2: MULTI-AXIS NOISE-FLOOR-ANCHORED MOTION SUPPRESSION (Requirement 3)
# ==============================================================================
def suppress_motion_rls(p_w, ax_w, ay_w, az_w, lags=(0, 2, 4, 6), sigma_floor_sq=4.0):
    """
    Projects out the linear subspace correlated with 3D acceleration across
    multiple delay taps (0, ~50, ~100, ~150 ms).
    Anchored to the physical accelerometer quantization floor (sigma_floor^2 = 4.0)
    so quiet resting jitter is never overfitted or subtracted from cardiac peaks.
    """
    N = len(p_w)
    cols = []
    for ch in (ax_w, ay_w, az_w):
        for k in lags:
            s_ch = np.roll(ch, k)
            if k > 0:
                s_ch[:k] = 0.0
            cols.append(s_ch)
    X = np.column_stack(cols)

    # Ridge-regularized least squares: w = (X^T X + N * sigma_floor^2 * I)^(-1) X^T p_w
    reg_matrix = (N * sigma_floor_sq) * np.eye(X.shape[1])
    w = np.linalg.solve(X.T @ X + reg_matrix, X.T @ p_w)
    motion_est = X @ w
    return p_w - motion_est, motion_est


# ==============================================================================
# STEP 3: SEGMENT PREPROCESSING, 10s HR ESTIMATION & ABSTENTION GATE (Req. 2 & 4)
# ==============================================================================
def run_hr_pipeline(csv_path, win_sec=12.0, step_sec=10.0, edge_trim_sec=2.0):
    # csv_path = input_file_path
    # win_sec = 12.0
    # step_sec = 10.0
    # edge_trim_sec = 2.0

    output_path = os.path.join(os.path.dirname(csv_path), os.path.splitext(os.path.basename(csv_path))[0])
    os.mkdir(output_path)

    print(f"Loading dataset: {csv_path} ...")
    df = pd.read_csv(csv_path)
    segments = extract_clean_segments(df, min_duration_s=win_sec + edge_trim_sec)
    print(f"Extracted {len(segments)} contiguous valid segments (>= {win_sec + edge_trim_sec}s).")

    all_windows = []

    for seg_idx, seg in enumerate(segments):

        # segment_path = os.path.join(output_path, str(seg_idx))
        # os.mkdir(segment_path)

        t = seg['elapsed_time_s'].values
        ppg_raw = seg['ppg'].values.astype(float)
        ax_raw = seg['acc_x'].values.astype(float)
        ay_raw = seg['acc_y'].values.astype(float)
        az_raw = seg['acc_z'].values.astype(float)

        # Empirical sampling rate across continuous span (~40.07 Hz)
        fs = (len(t) - 1) / (t[-1] - t[0])

        # 1. Derivative despiking: clamp single-sample optical gain/contact steps before IIR filter
        d_ppg = np.diff(ppg_raw, prepend=ppg_raw[0])
        med_d = np.median(d_ppg)
        mad_d = np.median(np.abs(d_ppg - med_d)) + 1e-6
        d_ppg_clipped = np.clip(d_ppg, med_d - 8.0 * mad_d, med_d + 8.0 * mad_d)
        ppg_despiked = np.cumsum(d_ppg_clipped)

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        # ax1.plot(t, ppg_raw)
        # ax2.plot(t, ppg_despiked)
        # ax1.set_title("RawPPG")
        # ax2.set_xlabel("Timestamp (s)")
        # ax2.set_title("Despiked PPG")
        # fig.savefig(os.path.join(segment_path, "Raw_vs_Despiked_PPG.png"), dpi=300, bbox_inches="tight")
        # # plt.show()
        # plt.close(fig)

        # 2. Zero-phase 4th-order Butterworth bandpass filter: [48 BPM (0.80 Hz), 200 BPM (3.33 Hz)]
        sos_bp = butter(4, [48.0 / 60.0, 200.0 / 60.0], btype='bandpass', fs=fs, output='sos')
        ppg_bp = sosfiltfilt(sos_bp, ppg_despiked)
        ax_bp = sosfiltfilt(sos_bp, ax_raw)
        ay_bp = sosfiltfilt(sos_bp, ay_raw)
        az_bp = sosfiltfilt(sos_bp, az_raw)

        # fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
        #
        # # PPG
        # axes[0, 0].plot(t, ppg_raw)
        # axes[0, 0].set_title("Raw PPG")
        #
        # axes[0, 1].plot(t, ppg_bp)
        # axes[0, 1].set_title("Filtered PPG")
        #
        # # ax
        # axes[1, 0].plot(t, ax_raw)
        # axes[1, 0].set_title("Raw ax")
        #
        # axes[1, 1].plot(t, ax_bp)
        # axes[1, 1].set_title("Filtered ax")
        #
        # # ay
        # axes[2, 0].plot(t, ay_raw)
        # axes[2, 0].set_title("Raw ay")
        #
        # axes[2, 1].plot(t, ay_bp)
        # axes[2, 1].set_title("Filtered ay")
        #
        # # az
        # axes[3, 0].plot(t, az_raw)
        # axes[3, 0].set_title("Raw az")
        #
        # axes[3, 1].plot(t, az_bp)
        # axes[3, 1].set_title("Filtered az")
        #
        #
        # axes[3, 0].set_xlabel("Timestamp (s)")
        # axes[3, 1].set_xlabel("Timestamp (s)")
        #
        # fig.tight_layout()
        #
        # fig.savefig(os.path.join(segment_path, "Raw_vs_Filtered_Input.png"), dpi=300, bbox_inches="tight")
        # # plt.show()
        # plt.close(fig)

        # 3. Trim first 2.0 seconds to eliminate filter turn-on settling transient
        trim_n = int(edge_trim_sec * fs)
        t_tr = t[trim_n:]
        ppg_tr = ppg_bp[trim_n:]
        ax_tr, ay_tr, az_tr = ax_bp[trim_n:], ay_bp[trim_n:], az_bp[trim_n:]

        # 4. Slide 12-second window stepped every 10 seconds
        cur_start = t_tr[0]
        prev_bpm = None

        # yes_check = True
        # no_check = True

        while cur_start + win_sec <= t_tr[-1] + 0.05:
            cur_end = cur_start + win_sec
            mask = (t_tr >= cur_start) & (t_tr < cur_end)

            p_w = ppg_tr[mask]
            ax_w, ay_w, az_w = ax_tr[mask], ay_tr[mask], az_tr[mask]

            # Clamp isolated intra-window pop outliers at 3x IQR before regression
            iqr_w = np.percentile(p_w, 75) - np.percentile(p_w, 25) + 1e-6
            p_w_clamped = np.clip(p_w, -3.0 * iqr_w, 3.0 * iqr_w)

            # Multi-axis motion artifact suppression
            p_clean, _ = suppress_motion_rls(p_w_clamped, ax_w, ay_w, az_w)
            acc_rms = np.sqrt(np.mean(ax_w**2 + ay_w**2 + az_w**2))

            # 5. Spectral HR Estimation (Welch PSD padded to 4096 FFT points)
            freqs, psd = welch(p_clean, fs=fs, window='hann', nperseg=len(p_clean), nfft=4096)
            cand_mask = (freqs >= 48.0 / 60.0) & (freqs <= 190.0 / 60.0)
            f_cand = freqs[cand_mask]
            psd_cand = psd[cand_mask]
            bpm_grid = f_cand * 60.0

            # Soft physiological continuity prior if a previous valid BPM exists in this segment
            if prev_bpm is not None:
                prior = 0.35 + 0.65 * np.exp(-0.5 * ((bpm_grid - prev_bpm) / 18.0) ** 2)
                scored_psd = psd_cand * prior
            else:
                scored_psd = psd_cand

            best_idx = np.argmax(scored_psd)
            f_dom = f_cand[best_idx]
            bpm_cand = f_dom * 60.0

            # 6. Signal Quality Index (SQI) 1: Spectral Concentration Ratio (SNR within +/- 9 BPM)
            peak_band = np.abs(f_cand - f_dom) <= 0.15
            spec_snr = np.sum(psd_cand[peak_band]) / (np.sum(psd_cand) + 1e-12)

            # 7. SQI 2: Normalized Autocorrelation Periodicity Peak at lag = fs / f_dom
            p_zero = p_clean - np.mean(p_clean)
            acf_full = np.correlate(p_zero, p_zero, mode='full')
            acf = acf_full[len(p_zero) - 1:] / (acf_full[len(p_zero) - 1] + 1e-12)
            lag0 = int(round(fs / f_dom))
            rad = max(2, int(round(0.12 * lag0)))
            l_min = max(1, lag0 - rad)
            l_max = min(len(acf) - 1, lag0 + rad)
            acf_peak = float(np.max(acf[l_min:l_max + 1]))

            # 8. Abstention Gate: Output NaN (blank) when signal quality is insufficient
            reasons = []
            if spec_snr < 0.30:
                reasons.append("LOW_SNR")
            if acf_peak < 0.20:
                reasons.append("LOW_ACF")
            if prev_bpm is not None and abs(bpm_cand - prev_bpm) > 22.0 and spec_snr < 0.45:
                reasons.append("SLEW_JUMP")

            is_valid = (len(reasons) == 0)
            bpm_reported = round(bpm_cand, 1) if is_valid else np.nan
            if is_valid:
                prev_bpm = bpm_cand

            all_windows.append({
                'segment_id': seg_idx,
                'window_start_s': round(cur_start, 2),
                'window_end_s': round(cur_end, 2),
                'report_time_s': round(cur_start + step_sec, 2),
                'bpm_estimate': bpm_reported,
                'bpm_candidate': round(bpm_cand, 1),
                'spec_snr': round(spec_snr, 3),
                'acf_peak': round(acf_peak, 3),
                'acc_rms': round(acc_rms, 3),
                'status': 'VALID' if is_valid else f"ABSTAIN ({','.join(reasons)})"
            })

            cur_start += step_sec

            # if is_valid:
            #     if yes_check:
            #         yes_check = False
            #
            #         subsegment_path = os.path.join(segment_path, 'Valid')
            #         os.mkdir(subsegment_path)
            #
            #         fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
            #         ax1.plot(t_tr[mask], p_w)
            #         ax1.set_title("PPG")
            #         ax2.plot(t_tr[mask], ax_w)
            #         ax2.set_title("ax")
            #         ax3.plot(t_tr[mask], ay_w)
            #         ax3.set_title("ay")
            #         ax4.plot(t_tr[mask], az_w)
            #         ax4.set_xlabel("Timestamp (s)")
            #         ax2.set_title("az")
            #         fig.savefig(os.path.join(subsegment_path, "Window signals.png"), dpi=300, bbox_inches="tight")
            #         # plt.show()
            #         plt.close(fig)
            #
            #         fig, ax = plt.subplots(figsize=(12, 5))
            #         ax.plot(f_cand, psd_cand)
            #         ax.set_xlabel("Frequency")
            #         ax.set_ylabel("PSD")
            #         ax.set_title("Power Spectral Density of PPG")
            #         ax.grid()
            #         fig.savefig(os.path.join(subsegment_path, "PSD.png"), dpi=300, bbox_inches="tight")
            #
            #         plt.close(fig)
            #         fig, ax = plt.subplots(figsize=(12, 5))
            #         ax.plot(t_tr[mask], p_w_clamped, label="Input PPG")
            #         ax.plot(t_tr[mask], p_clean, label="Output PPG")
            #         ax.set_xlabel("Timestamp (s)")
            #         ax.set_ylabel("Amplitude")
            #         ax.legend(loc='upper right', ncol=3, fontsize='small')
            #         ax.set_title("Motion Artifact Suppression")
            #         ax.grid()
            #         fig.savefig(os.path.join(subsegment_path, "Motion_Artifact_Suppression.png"), dpi=300, bbox_inches="tight")
            #         plt.close(fig)
            #
            # else:
            #     if no_check:
            #         if "LOW_SNR" in reasons:
            #             no_check = False
            #
            #             subsegment_path = os.path.join(segment_path, 'Invalid')
            #             os.mkdir(subsegment_path)
            #
            #             fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
            #             ax1.plot(t_tr[mask], p_w)
            #             ax1.set_title("PPG")
            #             ax2.plot(t_tr[mask], ax_w)
            #             ax2.set_title("ax")
            #             ax3.plot(t_tr[mask], ay_w)
            #             ax3.set_title("ay")
            #             ax4.plot(t_tr[mask], az_w)
            #             ax4.set_xlabel("Timestamp (s)")
            #             ax2.set_title("az")
            #             fig.savefig(os.path.join(subsegment_path, "Window signals.png"), dpi=300, bbox_inches="tight")
            #             # plt.show()
            #             plt.close(fig)
            #
            #             fig, ax = plt.subplots(figsize=(12, 5))
            #             ax.plot(f_cand, psd_cand)
            #             ax.set_xlabel("Frequency")
            #             ax.set_ylabel("PSD")
            #             ax.set_title("Power Spectral Density of PPG")
            #             ax.grid()
            #             fig.savefig(os.path.join(subsegment_path, "PSD.png"), dpi=300, bbox_inches="tight")
            #
            #             plt.close(fig)
            #             fig, ax = plt.subplots(figsize=(12, 5))
            #             ax.plot(t_tr[mask], p_w_clamped, label="Input PPG")
            #             ax.plot(t_tr[mask], p_clean, label="Output PPG")
            #             ax.set_xlabel("Timestamp (s)")
            #             ax.set_ylabel("Amplitude")
            #             ax.legend(loc='upper right', ncol=3, fontsize='small')
            #             ax.set_title("Motion Artifact Suppression")
            #             ax.grid()
            #             fig.savefig(os.path.join(subsegment_path, "Motion_Artifact_Suppression.png"), dpi=300,
            #                         bbox_inches="tight")
            #             plt.close(fig)



    results_df = pd.DataFrame(all_windows)

    # Save with blank ('') for NaN as required by Requirement 4 ("Use blank output when not trustworthy")
    results_df.to_csv(os.path.join(output_path, "BPM.csv"), index=False, na_rep='')
    return results_df


if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()

    input_file_path = filedialog.askopenfilename(
        title="Select the input CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    if not input_file_path:
        raise ValueError("No file selected.")

    if os.path.splitext(input_file_path)[1].lower() != ".csv":
        raise ValueError("Invalid file format. Please select a CSV file.")

    results_df = run_hr_pipeline(input_file_path)
