

# =====================================================================
# FILE: tornado.py
# Full-simulation one-way sensitivity analysis.
# =====================================================================
import copy
import numpy as np

from simulation import run_simulations

TORNADO_SPEC = {
    'efficacy': {'range': 0.2, 'clamp': (0, 1)},
    'cost_mud_mean': {'range': 0.3, 'clamp': (10, None)},
    'cost_brick_mean': {'range': 0.3, 'clamp': (10, None)},
    'max_coverage': {'range': 0.2, 'clamp': (0, 1)},
    'discount_rate': {'range': 0.5, 'clamp': (0, 0.2)},
    'failure_rate': {'range': 0.3, 'clamp': (0, 1)},
    'annual_repair_rate': {'range': 0.3, 'clamp': (0, 1)},
    'mal_prevalence': {'range': 0.25, 'clamp': (0.001, 1)},
    'modifiable_homes': {'range': 0.15, 'clamp': (0.1, 1)},
}


def _clamp(v, lo, hi):
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def set_param(p, rd, name, value):
    lo, hi = TORNADO_SPEC[name]['clamp']
    value = _clamp(value, lo, hi)
    if name == 'cost_mud_mean':
        p.cost_mud["mean"] = value
    elif name == 'cost_brick_mean':
        p.cost_brick["mean"] = value
    elif name in ('mal_prevalence', 'modifiable_homes'):
        rd[name] = value
    else:
        setattr(p, name, value)
    return p, rd


def _tornado_base_values(base_params, region_data):
    return {
        'efficacy': base_params.efficacy,
        'cost_mud_mean': base_params.cost_mud["mean"],
        'cost_brick_mean': base_params.cost_brick["mean"],
        'max_coverage': base_params.max_coverage,
        'discount_rate': base_params.discount_rate,
        'failure_rate': base_params.failure_rate,
        'annual_repair_rate': base_params.annual_repair_rate,
        'mal_prevalence': region_data["mal_prevalence"],
        'modifiable_homes': region_data["modifiable_homes"],
    }


def _nmb_from_sim(params, region_name, region_data, sim, n_simulations):
    results = run_simulations(params, {region_name: region_data},
                              n_simulations = n_simulations)
    cases = [s[0]["Cases_Averted"][-1] for s in results]
    costs = [sum(s[0]["Total_Costs"]) for s in results]
    return np.mean([(c * sim.threshold) - cost for c, cost in zip(cases, costs)])


def tornado_analysis(base_params, updated_region_data, region_name, sim, n_simulations=100):
    """Full-simulation NMB sensitivity analysis.
    Coverage adoption uses the UNVARIED selected region (updated_region_data),
    matching the original global-read behaviour. FLAG for recheck: decide
    whether varied mal_prevalence/modifiable_homes should also flow into the
    coverage adoption probability."""
    base_values = _tornado_base_values(base_params, updated_region_data)
    base_nmb = _nmb_from_sim(base_params, region_name, updated_region_data, sim,
                             n_simulations)

    tornado_results = []
    for name, base_value in base_values.items():
        variation = base_value * TORNADO_SPEC[name]['range']
        nmb = {}
        for tag, value in (("low", base_value - variation), ("high", base_value + variation)):
            p = copy.deepcopy(base_params)
            rd = copy.deepcopy(updated_region_data)
            set_param(p, rd, name, value)
            nmb[tag] = _nmb_from_sim(p, region_name, rd, sim, n_simulations)

        tornado_results.append({
            'parameter': name.replace('_', ' ').title(),
            'base_value': base_value,
            'low_value': base_value - variation,
            'high_value': base_value + variation,
            'low_nmb': nmb["low"],
            'high_nmb': nmb["high"],
            'nmb_range': abs(nmb["high"] - nmb["low"]),
            'base_nmb': base_nmb,
        })
    return tornado_results, base_nmb

