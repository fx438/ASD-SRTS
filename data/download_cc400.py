# -*- coding: utf-8 -*-
"""Download the ABIDE-I CC400 ROI time series used in the paper and convert them to the loader format.

Source: Preprocessed Connectomes Project (PCP), ABIDE Preprocessed, C-PAC pipeline, band-pass filtering
with global signal regression ("filt_global"), rois_cc400, served from the public fcp-indi S3 bucket.

Output layout (expected by experiments/dataset_cc400.py):

    data/ABIDE_CC400_Complete/
        Phenotypic.csv        ABIDE-I phenotypic table
        CSV/<FILE_ID>.csv     one row per volume, one column per ROI (392), no header

The 1035 FILE_IDs are listed in file_ids_1035.txt; after the T >= 146 filter the loader keeps the
860 participants listed in subjects_860.csv. Run:  python data/download_cc400.py  [--workers 8]
"""
import argparse
import csv
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative'
ROI_URL = BASE + '/Outputs/cpac/filt_global/rois_cc400/{}_rois_cc400.1D'
PHENO_URL = BASE + '/Phenotypic_V1_0b_preprocessed1.csv'
HERE = Path(__file__).resolve().parent


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise


def convert_1d(raw):
    """PCP .1D: one '#'-prefixed header line, tab-separated values -> comma-separated, no header."""
    rows = [line.split() for line in raw.decode('ascii').splitlines() if line.strip() and not line.startswith('#')]
    out = io.StringIO()
    csv.writer(out, lineterminator='\n').writerows(rows)
    return out.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=HERE / 'ABIDE_CC400_Complete')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    (args.out / 'CSV').mkdir(parents=True, exist_ok=True)

    pheno = args.out / 'Phenotypic.csv'
    if not pheno.exists():
        pheno.write_bytes(fetch(PHENO_URL))

    ids = [s.strip() for s in (HERE / 'file_ids_1035.txt').read_text().split() if s.strip()]

    def one(fid):
        dst = args.out / 'CSV' / f'{fid}.csv'
        if dst.exists() and dst.stat().st_size > 0:
            return None
        try:
            dst.write_text(convert_1d(fetch(ROI_URL.format(fid))), encoding='ascii')
            return None
        except Exception as exc:  # noqa: BLE001
            return f'{fid}: {exc}'

    with ThreadPoolExecutor(args.workers) as pool:
        failed = [r for r in pool.map(one, ids) if r]
    print(f'{len(ids) - len(failed)}/{len(ids)} subjects in {args.out / "CSV"}')
    if failed:
        print('failed:', *failed, sep='\n  ')
        sys.exit(1)


if __name__ == '__main__':
    main()
