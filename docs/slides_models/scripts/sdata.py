from sstyle import *
import pandas as pd, numpy as np, re
R = WS / 'landis_mogd' / 'results'
FOLDS = ['train', 'val', 'test']
MODELS = {  # key: (label, color, dir, filename pattern, rounds file)
 'bs':   ('Biomass Succession', GREEN, 'gafl', 'cv_heldout_{f}_groups.csv', 'final_calib_rounds.csv'),
 'pnet': ('PnET', SKY, 'pnet_gafl', {'train':'cvP_val_test_heldout_train_groups.csv','val':'cvP_train_test_heldout_val_groups.csv','test':'cvP_train_val_heldout_test_groups.csv'}, 'final_calib_rounds.csv'),
 'hbad': ('HBAD v1', AMBER, 'hbad_gafl', {'train':'cvH_val_test_heldout_train_groups.csv','val':'cvH_train_test_heldout_val_groups.csv','test':'cvH_train_val_heldout_test_groups.csv'}, 'final_calib_rounds.csv'),
 'simb': ('BS Sim B', RUST, 'gafl_simB', {'train':'cvB_val_test_heldout_train_groups.csv','val':'cvB_train_test_heldout_val_groups.csv','test':'cvB_train_val_heldout_test_groups.csv'}, 'final_calib_rounds_3lever.csv'),
}
def folds(key):
    lab, col, d, pat, rf = MODELS[key]
    out = {}
    for f in FOLDS:
        fn = pat.format(f=f) if isinstance(pat, str) else pat[f]
        out[f] = pd.read_csv(R / d / fn)
    return out
def rounds(key):
    lab, col, d, pat, rf = MODELS[key]
    return pd.read_csv(R / MODELS[key][2] / rf)
def pooled_n(fd):
    return pd.concat(fd.values()).groupby('group').n.sum().sort_values(ascending=False)
def pass_counts(fd):
    return {t: [int(df['pass_%d' % t].sum()) for df in fd.values()] for t in (5, 10, 20)}
