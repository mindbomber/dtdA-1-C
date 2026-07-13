#!/usr/bin/env python3
"""Corrections applied to the empirical cardiorespiratory study runner."""
import math
import re
import numpy as np
import pandas as pd
import wfdb
from empirical import run_empirical_cardiorespiratory as base


def parse_st_annotations(path, fs, duration_sec):
    """Combine multiple NOTE annotations that can occur in the same 30-second epoch."""
    ann = wfdb.rdann(path, 'st')
    n = int(math.ceil(duration_sec / 30))
    apnea = np.zeros(n, dtype=int)
    sleep = np.zeros(n, dtype=int)
    codes = []
    for sample, note in zip(ann.sample, ann.aux_note):
        i = int((sample / fs) // 30)
        if not 0 <= i < n:
            continue
        text = str(note).replace('\x00', ' ').strip().upper()
        tokens = set(re.findall(r'[A-Z]+|[1-4]', text))
        apnea[i] = max(apnea[i], int(bool(tokens & base.APNEA_CODES)))
        sleep[i] = max(sleep[i], int(bool(tokens & base.SLEEP_CODES)))
        codes.append(text)
    return apnea, sleep, codes


def topology_analysis(series):
    """Compare empirical R4 phase clouds with a genuine one-dimensional closed phase-locking null."""
    if base.ripser is None:
        return pd.DataFrame()
    chosen = sorted(series)[:5] + sorted(series)[-5:]
    rows = []
    for record in chosen:
        theta, phi = series[record]
        valid = np.where(np.isfinite(theta) & np.isfinite(phi))[0]
        if len(valid) < 200:
            continue
        idx = np.linspace(valid[0], valid[-1], 180).astype(int)
        theta = theta[idx]
        phi = phi[idx]
        locked_phi = np.mod(5 * theta, base.TWO_PI)
        clouds = {
            'empirical': np.column_stack([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]),
            'one_cycle_null': np.column_stack([np.cos(theta), np.sin(theta), np.cos(locked_phi), np.sin(locked_phi)]),
        }
        for kind, cloud in clouds.items():
            diagrams = base.ripser(cloud, maxdim=2, thresh=2.2)['dgms']
            p1 = np.sort((diagrams[1][:, 1] - diagrams[1][:, 0])[np.isfinite(diagrams[1][:, 1])])[::-1] if len(diagrams) > 1 else np.array([])
            p2 = np.sort((diagrams[2][:, 1] - diagrams[2][:, 0])[np.isfinite(diagrams[2][:, 1])])[::-1] if len(diagrams) > 2 else np.array([])
            rows.append({
                'record': record,
                'kind': kind,
                'h1_longest': p1[0] if len(p1) else 0,
                'h1_second': p1[1] if len(p1) > 1 else 0,
                'h2_longest': p2[0] if len(p2) else 0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(base.CSV / 'fantasia_topology.csv', index=False)
    return out


base.parse_st_annotations = parse_st_annotations
base.topology_analysis = topology_analysis

if __name__ == '__main__':
    base.main()
