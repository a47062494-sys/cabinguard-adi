"""Threshold tuning on the DEV split only (seeds disjoint from TEST)."""
import os, pickle, itertools, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from run_validation import _job, seeds, N, evaluate, summarise, EVENT_CLASSES
cache = "results/dev_cache.pkl"
if not os.path.exists(cache):
    jobs = [("normal", s, None) for s in seeds("dev", "normal", N["dev"]["normal"])]
    for c in EVENT_CLASSES: jobs += [(c, s, None) for s in seeds("dev", c, N["dev"]["event"])]
    with ProcessPoolExecutor(2) as ex: trials = list(ex.map(_job, jobs, chunksize=4))
    os.makedirs("results", exist_ok=True); pickle.dump(trials, open(cache, "wb"))
trials = pickle.load(open(cache, "rb"))
grid = dict(band_ratio=[1.5, 2.0, 2.5], energy_z=[1.5, 3.0], rhythm=[0.4, 0.45, 0.5], sz_verify_s=[1.5, 2.0])
print("band energy rhythm szv | sz_rate sz_lat mild gtcs_lat | FV prompts/h")
for vals in itertools.product(*grid.values()):
    P = dict(zip(grid, vals))
    S = summarise(evaluate(trials, P=P))
    sz = S["seizure"]; nm = S["normal"]
    print(vals, "|", round(sz["rate"], 2), round(sz["lat_median"] or -1, 1), sz["by_variant"]["mild"]["verified"], "/", sz["by_variant"]["mild"]["n"],
          round(sz["by_variant"]["gtcs"]["lat_median"], 1), "|", nm["false_verifications"], nm["preonset_false_verifications"], round(nm["prompts_per_h"], 1),
          "syn", S["syncope"]["verified"], "wrong", sum(S[c]["wrong_class"] for c in EVENT_CLASSES))
