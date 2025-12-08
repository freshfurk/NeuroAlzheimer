import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Teilnehmerdaten laden
# ---------------------------------------------------------

meta = pd.read_excel("data/participants_info.xlsx", sheet_name=0)
meta = meta[meta["Group"].isin(["A", "C"])]
meta["Label"] = meta["Group"].map({"A": 1, "C": 0})
meta.index = meta["participant_id"]

print("Teilnehmer geladen:")
print(meta.head())


# ---------------------------------------------------------
# 2. Funktion zum Finden der passenden .set-Datei
# ---------------------------------------------------------

def find_set_file(pid, label):
    folder = "alzheimer" if label == 1 else "healthy"
    folder_path = f"data/{folder}"

    for file in os.listdir(folder_path):
        if file.startswith(pid) and file.endswith("_task-eyesclosed_eeg.set"):
            return os.path.join(folder_path, file)
    return None


# ---------------------------------------------------------
# 3. Beispiel EEG zeigen (original + gefiltert)
# ---------------------------------------------------------

def show_raw_and_filtered(pid):
    row = meta.loc[pid]
    path = find_set_file(pid, row["Label"])

    raw = mne.io.read_raw_eeglab(path, preload=True)
    print(f"Zeige EEG für {pid}")

    # Rohsignal (5s)
    raw.copy().crop(tmin=0, tmax=5).plot(title=f"{pid} – Roh-EEG 0–5s")

    # Gefiltertes Signal
    raw_filt = raw.copy().filter(1, 40).crop(tmin=0, tmax=5)
    raw_filt.plot(title=f"{pid} – Gefiltert 1–40 Hz")

# Beispiel:
show_raw_and_filtered("sub-001")  # Alzheimer
show_raw_and_filtered("sub-037")  # Healthy

# ---------------------------------------------------------
# 4. Moderne PSD-API + Feature-Extraktion
# ---------------------------------------------------------

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}

def extract_features(raw):
    raw_f = raw.copy().filter(1, 40)

    # 2-Sekunden-Epochen erzeugen (2s × 500 Hz = 1000 Samples)
    epochs = mne.make_fixed_length_epochs(raw_f, duration=2.0, preload=True)

    # PSD mit korrekten Parametern
    psd = epochs.compute_psd(
        method="welch",
        fmin=1,
        fmax=30,
        n_fft=1000,        # gleich lang wie Epochengröße
        n_per_seg=1000     # Zero-padding erlaubt
    )

    freqs = psd.freqs
    psd_data = psd.get_data()  # (epochs, channels, freqs)
    psd_mean = psd_data.mean(axis=(0, 1))  # über alle Kanäle/Epochen mitteln

    feats = {}
    for band, (f1, f2) in FREQ_BANDS.items():
        mask = (freqs >= f1) & (freqs <= f2)
        feats[f"{band}_power"] = np.trapz(psd_mean[mask], freqs[mask])

    # Peak Frequency im Alpha-Bereich
    mask = (freqs >= 6) & (freqs <= 14)
    feats["peak_freq"] = freqs[mask][np.argmax(psd_mean[mask])]

    return feats, freqs, psd_mean

# ---------------------------------------------------------
# 5. Alle Daten verarbeiten
# ---------------------------------------------------------

X = []
y = []
psd_alz = []
psd_healthy = []

for pid, row in meta.iterrows():
    path = find_set_file(pid, row["Label"])
    if path is None:
        print("Fehlt:", pid)
        continue

    print("Lade:", pid, "→", path)
    raw = mne.io.read_raw_eeglab(path, preload=True)

    feats, freqs, psd_mean = extract_features(raw)
    X.append(list(feats.values()))
    y.append(row["Label"])

    if row["Label"] == 1:
        psd_alz.append(psd_mean)
    else:
        psd_healthy.append(psd_mean)

X = np.array(X)
y = np.array(y)


# ---------------------------------------------------------
# 6. Klassifikation Alzheimer vs Healthy
# ---------------------------------------------------------

clf = SVC(kernel="rbf", gamma="scale")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n===== KLASSIFIKATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nKonfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))


# ---------------------------------------------------------
# 7. Durchschnittliche PSD Alzheimer vs Healthy
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(freqs, np.mean(psd_healthy, axis=0), label="Healthy", color="green")
plt.plot(freqs, np.mean(psd_alz, axis=0), label="Alzheimer", color="red")
plt.title("Durchschnittliche PSD – Alzheimer vs Healthy")
plt.xlabel("Frequenz (Hz)")
plt.ylabel("Power")
plt.grid()
plt.legend()
plt.show()


# ---------------------------------------------------------
# 8. MMSE vs Alpha-Power Diagramm
# ---------------------------------------------------------

alpha_index = list(FREQ_BANDS.keys()).index("alpha")
alpha_power = X[:, alpha_index]

plt.figure(figsize=(8, 5))
plt.scatter(meta["MMSE"], alpha_power, c=y, cmap="coolwarm")
plt.xlabel("MMSE")
plt.ylabel("Alpha-Power")
plt.title("MMSE vs Alpha-Power")
plt.grid()
plt.colorbar(label="0 = Healthy, 1 = Alzheimer")
plt.show()

# ---------------------------------------------------------
# 9. Beispiel: Epochen visualisieren (Segmentierung)
# ---------------------------------------------------------

sample_pid = meta.index[0]
raw_example = mne.io.read_raw_eeglab(find_set_file(sample_pid, meta.loc[sample_pid]["Label"]), preload=True)
raw_example_f = raw_example.copy().filter(1, 40)
epochs_example = mne.make_fixed_length_epochs(raw_example_f, duration=2.0, preload=True)

epochs_example.plot(title="Epochenansicht: 2-Sekunden-Segmente")


# ---------------------------------------------------------
# 10. Welch-PSD Darstellung für ein Beispiel
# ---------------------------------------------------------

psd_ex = epochs_example.compute_psd(method="welch", fmin=1, fmax=30, n_fft=1000, n_per_seg=1000)
freqs_ex = psd_ex.freqs
psd_mean_ex = psd_ex.get_data().mean(axis=(0, 1))

plt.figure(figsize=(8, 5))
plt.semilogy(freqs_ex, psd_mean_ex)
plt.xlabel("Frequenz (Hz)")
plt.ylabel("PSD")
plt.title("PSD eines Beispiel-Teilnehmers")
plt.grid()
plt.show()


# ---------------------------------------------------------
# 11. Bandpower-Diagramm (mittlerer Vergleich)
# ---------------------------------------------------------

band_means_alz = np.mean(X[y == 1, :4], axis=0)
band_means_healthy = np.mean(X[y == 0, :4], axis=0)

labels = list(FREQ_BANDS.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width, band_means_healthy, width, label="Healthy", color="green")
plt.bar(x + width, band_means_alz, width, label="Alzheimer", color="red")
plt.xticks(x, labels)
plt.ylabel("Power")
plt.title("Durchschnittliche Bandpower")
plt.legend()
plt.grid(axis="y")
plt.show()


# ---------------------------------------------------------
# 12. Peak Frequency (Alpha) visualisieren
# ---------------------------------------------------------

peak_freqs = X[:, -1]  # letzte Spalte im Feature-Vektor

plt.figure(figsize=(8, 5))
plt.hist(peak_freqs[y == 0], bins=15, alpha=0.6, label="Healthy", color="green")
plt.hist(peak_freqs[y == 1], bins=15, alpha=0.6, label="Alzheimer", color="red")
plt.xlabel("Peak Frequency (Hz)")
plt.ylabel("Anzahl")
plt.title("Peak Frequency im Alpha-Bereich")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------
# 13. Scatter-Plot der Klassifikation (ML)
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, marker="o", label="Train")
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, marker="x", label="Test")
plt.xlabel("Alpha-Power")
plt.ylabel("Beta-Power")
plt.title("SVM Klassifikation: Features Alpha vs Beta")
plt.legend()
plt.grid()
plt.show()
