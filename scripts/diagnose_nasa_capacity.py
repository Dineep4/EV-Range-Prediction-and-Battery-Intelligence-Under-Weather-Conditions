"""Inspect raw NASA discharge capacity values to diagnose suspicious SOH baselines.
Run from project root: python scripts/diagnose_nasa_capacity.py
This script only reads raw files and prints diagnostics; it does not alter data.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'NASA'


def field(obj, name, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def items(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def scalar(value):
    try:
        a = np.asarray(value, dtype=float).squeeze()
        if a.size:
            v = float(a.reshape(-1)[0])
            return v if np.isfinite(v) else float('nan')
    except (TypeError, ValueError, OverflowError):
        pass
    return float('nan')


def arrlen(value):
    try:
        return int(np.asarray(value).size)
    except Exception:
        return 0


def main():
    paths = sorted(RAW.glob('*.mat'))
    if not paths:
        raise SystemExit(f'No .mat files found in {RAW}')
    for path in paths:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat if not k.startswith('__')]
        battery = next((mat[k] for k in keys if field(mat[k], 'cycle') is not None), mat[keys[0]])
        cycles = items(field(battery, 'cycle'))
        discharge = []
        for source_idx, cy in enumerate(cycles, 1):
            kind = field(cy, 'type', '')
            if isinstance(kind, bytes):
                kind = kind.decode(errors='replace')
            if str(kind).strip().lower() != 'discharge':
                continue
            data = field(cy, 'data')
            cap = scalar(field(data, 'Capacity'))
            discharge.append({
                'source_idx': source_idx,
                'cap': cap,
                'voltage_n': arrlen(field(data, 'Voltage_measured')),
                'current_n': arrlen(field(data, 'Current_measured')),
                'temp_n': arrlen(field(data, 'Temperature_measured')),
                'time_n': arrlen(field(data, 'Time')),
            })
        suspicious = [r for r in discharge if not np.isfinite(r['cap']) or r['cap'] <= 0 or r['cap'] < 0.5]
        print(f'\n{path.stem}: {len(discharge)} discharge cycles; first 8 capacities:')
        for r in discharge[:8]:
            print('  source_cycle={source_idx:>3} capacity={cap:.6f} Ah samples(V/I/T/time)={voltage_n}/{current_n}/{temp_n}/{time_n}'.format(**r))
        if suspicious:
            print(f'  LOW/INVALID (<0.5 Ah or invalid): {len(suspicious)} cycle(s)')
            for r in suspicious[:12]:
                print('    source_cycle={source_idx:>3} capacity={cap:.6f} Ah samples(V/I/T/time)={voltage_n}/{current_n}/{temp_n}/{time_n}'.format(**r))
            if len(suspicious) > 12:
                print(f'    ... plus {len(suspicious)-12} more')

if __name__ == '__main__':
    main()
