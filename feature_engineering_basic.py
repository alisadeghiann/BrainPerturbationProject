from pathlib import Path
import mne
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
INPUT = BASE / "final_dataset" / "perturbation" / "epochs"
OUTPUT = BASE / "features" / "basic"
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.glob("*_final_epo.fif"))

print("=" * 80)
print("FEATURE ENGINEERING - PSD + BAND POWER")
print("=" * 80)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

all_features = []

for i, f in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {f.name}")

    epochs = mne.read_epochs(f, preload=True, verbose=False)

    picks = mne.pick_types(
        epochs.info,
        eeg=True,
        eog=False,
        exclude="bads"
    )

    data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]

    # PSD: epochs x channels x frequencies
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=30,
        n_fft=256,
        verbose=False
    )

    # Safety: make sure PSD is 3D
    if psd.ndim == 2:
        psd = psd[np.newaxis, :, :]

    channel_names = np.array(epochs.ch_names)[picks]

    for band, (low, high) in BANDS.items():

        mask = (freqs >= low) & (freqs < high)

        band_power = psd[:, :, mask].mean(axis=-1)

        for epoch_idx in range(band_power.shape[0]):

            for ch_idx, ch_name in enumerate(channel_names):

                all_features.append({
                    "file": f.name,
                    "subject": f.name.split("_")[0],
                    "run": f.name.split("_")[1],
                    "epoch": epoch_idx,
                    "channel": ch_name,
                    "band": band,
                    "power": float(band_power[epoch_idx, ch_idx]),
                })

df = pd.DataFrame(all_features)

output_file = OUTPUT / "eeg_band_power_features.csv"
df.to_csv(output_file, index=False)

print()
print("=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print(f"Runs:       {len(files)}")
print(f"Rows:       {len(df):,}")
print(f"Channels:   {df['channel'].nunique()}")
print(f"Bands:      {df['band'].nunique()}")
print(f"Subjects:   {df['subject'].nunique()}")

print()
print("Saved:")
print(output_file)

print("=" * 80)
