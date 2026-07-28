
# =====================================================================
# FILE: economics.py
# Cost-effectiveness (CEAC) and savings-column construction.
# =====================================================================
import numpy as np

SUFFIXES = ["", "_lower", "_upper", "_cum", "_cum_lower", "_cum_upper"]


def compute_ceac(cost_samples, daly_samples, wtp_range):
    """ICER draws computed once, compared against each WTP threshold."""
    cost = np.asarray(cost_samples, dtype=float)
    daly = np.asarray(daly_samples, dtype=float)
    mask = daly != 0
    ratios = cost[mask] / daly[mask]
    ceac = [float(np.mean(ratios < t)) for t in wtp_range]
    return ceac, ratios


def add_savings_columns(df, sim):
    transfusion = sim.transfusion_cost * sim.prop_sam_transfused
    for s in SUFFIXES:
        df[f"hs_savings{s}"] = (
            df[f"opd_cost{s}"] + df[f"ipd_cost{s}"]
            + df[f"severe_anemia{s}"] * (transfusion + sim.hematinics)
        )
    name_map = {
        "": "_no_death", "_lower": "_no_death_lower", "_upper": "_no_death_upper",
        "_cum": "_cum_no_death", "_cum_lower": "_cum_no_death_lower",
        "_cum_upper": "_cum_no_death_upper",
    }
    for src, tgt in name_map.items():
        base = df[f"hs_savings{src}"] + df[f"um_hh_cost{src}"] + df[f"sm_hh_cost{src}"]
        df[f"societal_savings{tgt}"] = base
        df[f"societal_savings{tgt.replace('no_death', 'death')}"] = base + df[f"vsl{src}"]
    return df

