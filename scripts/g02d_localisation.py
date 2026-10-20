"""Localisation table for the generative-vegetation likelihood (rerun after every g02b rebuild). g02b computes, on the FULL disk of footprints (before
the candidate subsample): n_hi = footprints with prior >= 4 x median prior, mass_hi = their prior mass, n_fp = footprints; and flags the kept
candidates that belong to that set (cands.hi, used by g04.hi_cands). This copies those per-plot numbers with the FIA plot identity (st, u, c, p) next
to pid: pid is a row number of the current table and shifts when the visit set changes, so anything stored on disk is keyed by the plot identity
(g02c file names). Writes data/proto/gafl/localisation.parquet (index pid; columns n_hi, mass_hi, n, st, u, c, p)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT
D = f'{ROOT}/data/proto/gafl'
P = pd.read_parquet(f'{D}/plots.parquet'); P = P[P.n_fp.notna()].set_index('pid')
S = P[['n_hi', 'mass_hi', 'n_fp', 'st', 'u', 'c', 'p']].rename(columns={'n_fp': 'n'})
S.to_parquet(f'{D}/localisation.parquet')
sel = S[(S.n_hi > 0) & (S.n_hi <= 50) & (S.mass_hi >= 0.5)]
print(f'plots {len(S)}; localised (0 < n_hi <= 50 footprints, mass_hi >= 0.5): {len(sel)}')
