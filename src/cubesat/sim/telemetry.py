"""Telemetry logging to CSV."""

import csv


def write_csv(path, data):
    """Write a dict of equal-length arrays/sequences to ``path`` as CSV."""
    keys = list(data.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            writer.writerow(row)