"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    # One 12-hour mean per patient
    patient_means = readings.mean(axis=1)

    # Whole-file answers
    patients = len(patient_ids)
    n_readings = readings.size
    mean_sbp = readings.mean()
    sd_sbp = readings.std()
    min_sbp = int(readings.min())
    max_sbp = int(readings.max())

    # Patients with a 12-hour mean of 140 mmHg or higher
    stage2_patients = int((patient_means >= 140).sum())

    # Patient with the highest 12-hour mean
    top = patient_means.argmax()
    highest_patient = str(patient_ids[top])
    highest_patient_mean = patient_means[top]

    # Hour column with the highest mean across all patients
    hour_means = readings.mean(axis=0)
    peak = hour_means.argmax()
    peak_hour_column = str(hour_columns[peak])
    peak_hour_mean = hour_means[peak]

    # Each monitor's average, in the order of sorted(set(monitors))
    monitor_ids = sorted(set(monitors))
    monitor_avgs = []
    for m in monitor_ids:
        mask = monitors == m
        monitor_avgs.append(patient_means[mask].mean())
    monitor_avgs = np.array(monitor_avgs)

    # Highest monitor, and how far above the other monitors' patients it sits
    high_monitor = str(monitor_ids[monitor_avgs.argmax()])
    high_mask = monitors == high_monitor
    others_mean = patient_means[~high_mask].mean()
    monitor_offset = monitor_avgs.max() - others_mean
    stage2_other_monitors = int((patient_means[~high_mask] >= 140).sum())

    # Write one "key: value" line per answer; "w" replaces the file on each run
    with open("output/vitals_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"patients: {patients}\n")
        file.write(f"readings: {n_readings}\n")
        file.write(f"mean_sbp: {mean_sbp:.2f}\n")
        file.write(f"sd_sbp: {sd_sbp:.2f}\n")
        file.write(f"min_sbp: {min_sbp}\n")
        file.write(f"max_sbp: {max_sbp}\n")
        file.write(f"stage2_patients: {stage2_patients}\n")
        file.write(f"highest_patient: {highest_patient}\n")
        file.write(f"highest_patient_mean: {highest_patient_mean:.2f}\n")
        file.write(f"peak_hour_column: {peak_hour_column}\n")
        file.write(f"peak_hour_mean: {peak_hour_mean:.2f}\n")
        file.write(f"high_monitor: {high_monitor}\n")
        file.write(f"monitor_offset: {monitor_offset:.2f}\n")
        file.write(f"stage2_other_monitors: {stage2_other_monitors}\n")


if __name__ == "__main__":
    main()