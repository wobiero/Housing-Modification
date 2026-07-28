# =====================================================================
# FILE: policy.py
# Improved subsidy-policy engine for the Housing Modification Economic
# Analysis app.
#
# Key design choices:
#   1. A subsidy is a transfer, not a reduction in real resource cost.
#   2. Adoption is driven by the household-facing price using an
#      interpretable, two-point calibrated logistic curve.
#   3. Budget caps are national and annual, not per-region lifetime caps.
#   4. If a cap binds, households who would have adopted without subsidy
#      can still adopt at the full price; the cap only limits subsidized
#      vouchers and induced additional adoption.
#   5. DALYs are computed from deaths and morbidity per averted case,
#      rather than treating every malaria case averted as a death.
#   6. Budget impact tables use undiscounted cash outlays; CEA/NMB uses
#      discounted costs, savings, and effects.
# =====================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.special import expit, logit

USD_TO_UGX = 3594.0  # retained for legacy adoption_probability compatibility
EPS = 1e-9


# ---------------------------------------------------------------------
# Policy and adoption definitions
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SubsidyPolicy:
    name: str
    rate: float = 0.0                         # subsidy share of resource cost, 0..1
    target: str = "universal"                # universal | poor | high_transmission | hybrid
    poverty_var: str = "mdp"                 # mdp | subsistence | pdm
    transmission_threshold: float = 0.10      # prevalence cutoff for high-transmission targeting
    pass_through: float = 1.0                 # share of subsidy reaching household price
    admin_cost_per_hh_usd: float = 0.0        # per subsidized household, not all households
    annual_budget_cap_usd: Optional[float] = None
    allocation_rule: str = "proportional"    # proportional | poor_first | high_transmission_first | cea_first
    repair_subsidy_rate: float = 0.0          # default: households pay repairs

    def base_rate(self) -> float:
        """Subsidy for the non-targeted share."""
        return 0.0


@dataclass(frozen=True)
class AdoptionCalibration:
    """Interpretable adoption curve.

    p_full_price is the adoption probability at reference_price_usd.
    p_free is the adoption probability when household price is zero.

    If reference_price_usd is None, the model uses the national weighted
    average resource cost implied by params and the input regions.
    """
    p_full_price: float = 0.20
    p_free: float = 0.85
    reference_price_usd: Optional[float] = None
    poor_price_multiplier: float = 0.0        # 0.5 means poor households perceive price as 50% higher
    min_probability: float = 0.0
    max_probability: float = 1.0


def _clip01(x: float, eps: float = 1e-6) -> float:
    return min(max(float(x), eps), 1.0 - eps)


def _safe_rate(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def national_reference_price(params, regions: Dict[str, dict]) -> float:
    """Weighted average full resource price among modifiable households."""
    num, den = 0.0, 0.0
    for r in regions.values():
        mod_hh = float(r["households"]) * float(r["modifiable_homes"])
        price = (params.cost_mud["mean"] * float(r["mud"]) +
                 params.cost_brick["mean"] * float(r["bricks"]))
        num += mod_hh * price
        den += mod_hh
    return num / den if den > 0 else 1.0


def calibrated_adoption_probability(
    household_price_usd: float,
    calibration: AdoptionCalibration,
    reference_price_usd: float,
    is_poor: bool = False,
) -> float:
    """Adoption probability from a two-point calibrated logistic curve."""
    ref = max(float(reference_price_usd), EPS)
    p_free = _clip01(calibration.p_free)
    p_ref = _clip01(calibration.p_full_price)

    # Make the curve monotonically decreasing in price. If a slider setting
    # violates this, nudge p_ref just below p_free rather than crashing the app.
    if p_ref >= p_free:
        p_ref = max(p_free - 1e-4, 1e-6)

    alpha = logit(p_free)
    beta = (logit(p_ref) - alpha) / ref

    effective_price = max(float(household_price_usd), 0.0)
    if is_poor:
        effective_price *= (1.0 + max(float(calibration.poor_price_multiplier), 0.0))

    p = float(expit(alpha + beta * effective_price))
    return float(np.clip(p, calibration.min_probability, calibration.max_probability))


def adoption_probability(
    household_price_usd,
    params=None,
    alpha=134.37329,
    beta_price=-0.000336,
    *,
    calibration: Optional[AdoptionCalibration] = None,
    reference_price_usd: Optional[float] = None,
    is_poor: bool = False,
):
    """Backward-compatible adoption function.

    Old calls, adoption_probability(price, params), still use the legacy
    UGX-based coefficients. New policy simulations pass AdoptionCalibration.
    """
    if calibration is None:
        price_ugx = float(household_price_usd) * USD_TO_UGX
        return float(expit(alpha + beta_price * price_ugx))
    if reference_price_usd is None:
        raise ValueError("reference_price_usd is required with AdoptionCalibration")
    return calibrated_adoption_probability(
        household_price_usd, calibration, reference_price_usd, is_poor=is_poor)


def adoption_probability_affordable(household_price_usd, consumption_usd,
                                    alpha, beta_price, poverty=0.0, beta_pov=0.0):
    """Optional affordability form for future household-consumption data."""
    ratio = household_price_usd / max(consumption_usd, EPS)
    return float(expit(alpha + beta_price * np.log(ratio) + beta_pov * poverty))


# ---------------------------------------------------------------------
# Targeting groups and financial flows
# ---------------------------------------------------------------------
def region_groups(region_data: dict, policy: SubsidyPolicy) -> List[Tuple[float, float, bool]]:
    """Return (population share, subsidy_rate, is_poor) groups.

    We split universal and high-transmission policies by poverty status too,
    so equity metrics such as poor households reached are meaningful.
    """
    poor_share = _clip01(region_data.get(policy.poverty_var, 0.0), eps=0.0)
    nonpoor_share = max(1.0 - poor_share, 0.0)
    high_tx = float(region_data["mal_prevalence"]) >= float(policy.transmission_threshold)

    target = policy.target.lower().strip()
    if target == "universal":
        poor_rate = nonpoor_rate = policy.rate
    elif target == "high_transmission":
        poor_rate = nonpoor_rate = policy.rate if high_tx else policy.base_rate()
    elif target == "poor":
        poor_rate = policy.rate
        nonpoor_rate = policy.base_rate()
    elif target in {"hybrid", "hybrid_or", "poor_or_high_transmission"}:
        # Union rule: poor households everywhere, plus all households in
        # high-transmission regions.
        poor_rate = policy.rate
        nonpoor_rate = policy.rate if high_tx else policy.base_rate()
    elif target in {"hybrid_and", "poor_high_transmission", "poor_and_high_transmission"}:
        # Intersection rule: only poor households in high-transmission regions.
        poor_rate = policy.rate if high_tx else policy.base_rate()
        nonpoor_rate = policy.base_rate()
    else:
        raise ValueError(f"unknown target: {policy.target}")

    groups = []
    if poor_share > 0:
        groups.append((poor_share, _safe_rate(poor_rate), True))
    if nonpoor_share > 0:
        groups.append((nonpoor_share, _safe_rate(nonpoor_rate), False))
    return groups


def financial_flows(resource_cost: float, subsidy_rate: float, policy: SubsidyPolicy) -> dict:
    """Per-household flow decomposition.

    resource_cost is the full real economic cost. The transfer is tracked
    separately and is not counted as a societal resource cost.
    """
    s = _safe_rate(subsidy_rate)
    rho = min(max(float(policy.pass_through), 0.0), 1.0)
    admin = float(policy.admin_cost_per_hh_usd) if s > 0 else 0.0

    price_reduction = resource_cost * rho * s
    hh_price = max(resource_cost - price_reduction, 0.0)
    gov_transfer = resource_cost * s
    supplier_capture = gov_transfer - price_reduction

    return {
        "household_price": hh_price,
        "government_transfer": gov_transfer,
        "admin": admin,
        "government_outlay": gov_transfer + admin,
        "supplier_capture": supplier_capture,
        "societal_resource_cost": resource_cost + admin,
    }


# ---------------------------------------------------------------------
# Random draws
# ---------------------------------------------------------------------
def _gamma_draw(mean: float, sd: float, rng: Optional[np.random.Generator] = None) -> float:
    if mean <= 0 or sd <= 0:
        return float(mean)
    shape = (mean / sd) ** 2
    scale = (sd ** 2) / mean
    if rng is None:
        return float(np.random.gamma(shape, scale))
    return float(rng.gamma(shape, scale))


def _beta_draw(mean: float, sd: float, rng: np.random.Generator) -> float:
    mean = _clip01(mean)
    max_sd = np.sqrt(mean * (1.0 - mean))
    sd = min(max(float(sd), 1e-8), max_sd - 1e-8)
    var = sd ** 2
    nu = mean * (1.0 - mean) / var - 1.0
    a = mean * nu
    b = (1.0 - mean) * nu
    return float(rng.beta(a, b))


def draw_resource_costs(params, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    return (_gamma_draw(params.cost_mud["mean"], params.cost_mud["std"], rng),
            _gamma_draw(params.cost_brick["mean"], params.cost_brick["std"], rng))


def draw_savings_per_case(sim, rng: np.random.Generator) -> Tuple[float, float]:
    """Discount-compatible treatment savings per averted case.

    This mirrors the rest of the app's savings construction: outpatient +
    inpatient costs per averted case, plus severe-anemia-specific blood and
    haematinic costs; household costs are uncomplicated household cost plus
    severe-malaria household cost weighted by severe share.
    """
    opd = _gamma_draw(sim.opd_cost, sim.opd_cost_sd, rng)
    ipd = _gamma_draw(sim.ipd_cost, sim.ipd_cost_sd, rng)
    hh_um = _gamma_draw(sim.hh_um_cost, sim.hh_um_cost_sd, rng)
    hh_sm = _gamma_draw(sim.hh_sm_cost, sim.hh_sm_cost_sd, rng)

    severe = _beta_draw(sim.severe_mean, sim.severe_sd, rng)
    severe_anemia = _beta_draw(sim.severe_anemia_mean, sim.severe_anemia_sd, rng)
    transfusion = sim.transfusion_cost * sim.prop_sam_transfused

    hs_per_case = opd + ipd + severe_anemia * (transfusion + sim.hematinics)
    hh_per_case = hh_um + severe * hh_sm
    return hs_per_case, hh_per_case


def draw_daly_per_case(sim, rng: np.random.Generator) -> float:
    """DALYs averted per malaria case averted.

    This uses the same ingredients as the regional DALY section: morbidity
    duration/disability weights plus death probability times discounted life
    expectancy. It avoids the previous shortcut of applying disc_lifespan to
    every case averted.
    """
    death_p = _beta_draw(sim.deaths, sim.deaths_sd, rng)
    severe_p = _beta_draw(sim.severe_mean, sim.severe_sd, rng)
    severe_anemia_p = _beta_draw(sim.severe_anemia_mean, sim.severe_anemia_sd, rng)
    cerebral_p = _beta_draw(sim.cerebral_malaria_mean, sim.cerebral_malaria_sd, rng)
    cerebral_anemia_p = _beta_draw(sim.cerebral_anemia, sim.cerebral_anemia_sd, rng)

    um_dur = _gamma_draw(sim.duration_um, sim.duration_um_sd, rng)
    sm_dur = _gamma_draw(sim.duration_sm, sim.duration_sm_sd, rng)
    sa_dur = _gamma_draw(sim.duration_severe_anemia, sim.duration_severe_anemia_sd, rng)
    cer_dur = _gamma_draw(sim.duration_cerebral, sim.duration_cerebral_sd, rng)

    dw_um = _beta_draw(sim.dw_um, sim.dw_um_sd, rng)
    dw_sm = _beta_draw(sim.dw_sm, sim.dw_sm_sd, rng)
    dw_sa = _beta_draw(sim.dw_sev_anemia, sim.dw_sev_anemia_sd, rng)
    dw_cer = _beta_draw(sim.dw_cerebral, sim.dw_cerebral_sd, rng)
    dw_cer_an = _beta_draw(sim.dw_cerebral_anemia, sim.dw_cerebral_anemia_sd, rng)

    uncomplicated = um_dur * dw_um / 365.0
    severe = severe_p * sm_dur * dw_sm / 365.0
    severe_anemia = severe_anemia_p * sa_dur * dw_sa / 365.0
    cerebral = cerebral_p * cer_dur * dw_cer / 365.0
    cerebral_anemia = cerebral_anemia_p * cer_dur * dw_cer_an / 365.0
    death = death_p * sim.disc_lifespan

    return float(uncomplicated + severe + severe_anemia + cerebral + cerebral_anemia + death)


# ---------------------------------------------------------------------
# Coverage and stock mechanics
# ---------------------------------------------------------------------
def coverage_curve(years, prob_adopt: float, params) -> np.ndarray:
    denom = (1 + np.exp(-params.growth_rate * (years - params.midpoint_year))) ** params.shape
    return params.max_coverage * prob_adopt / denom


def _new_houses_from_probability(group_hh: float, prob_adopt: float, params) -> np.ndarray:
    coverage = coverage_curve(params.years, prob_adopt, params)
    new_fraction = np.diff(coverage, prepend=0)
    return np.maximum(new_fraction, 0.0) * group_hh


def _functional_stock_and_repairs(new_houses: np.ndarray, params) -> Tuple[np.ndarray, np.ndarray]:
    """Track functioning modified houses and annual repairs.

    A share of functioning houses fail each year. A share of the failed stock
    is repaired each year. Repair costs are charged on repairs, not on the
    accumulated probability of ever requiring repair.
    """
    n = len(new_houses)
    functional = np.zeros(n)
    repairs = np.zeros(n)
    failed_stock = 0.0

    for t in range(n):
        previous_functional = functional[t - 1] if t > 0 else 0.0
        failed_this_year = previous_functional * float(params.failure_rate)
        repaired_this_year = failed_stock * float(params.annual_repair_rate)
        failed_stock = max(failed_stock + failed_this_year - repaired_this_year, 0.0)
        functional[t] = max(previous_functional - failed_this_year + repaired_this_year + new_houses[t], 0.0)
        repairs[t] = repaired_this_year

    return functional, repairs


# ---------------------------------------------------------------------
# Simulation internals
# ---------------------------------------------------------------------
def _make_group_records(
    params,
    regions: Dict[str, dict],
    policy: SubsidyPolicy,
    cost_mud: float,
    cost_brick: float,
    modifiable_samples: Dict[str, float],
    adoption: AdoptionCalibration,
    reference_price_usd: float,
) -> List[dict]:
    records = []
    for region_name, r in regions.items():
        avg_resource = cost_mud * float(r["mud"]) + cost_brick * float(r["bricks"])
        modifiable_hh_total = float(r["households"]) * float(modifiable_samples[region_name])

        for share, s_rate, is_poor in region_groups(r, policy):
            if share <= 0:
                continue
            group_hh = modifiable_hh_total * share
            flows = financial_flows(avg_resource, s_rate, policy)
            flows0 = financial_flows(avg_resource, policy.base_rate(), policy)

            p_adopt = calibrated_adoption_probability(
                flows["household_price"], adoption, reference_price_usd, is_poor=is_poor)
            p_adopt0 = calibrated_adoption_probability(
                flows0["household_price"], adoption, reference_price_usd, is_poor=is_poor)

            desired_new = _new_houses_from_probability(group_hh, p_adopt, params)
            baseline_new = _new_houses_from_probability(group_hh, p_adopt0, params)

            records.append({
                "region_name": region_name,
                "region_data": r,
                "share": share,
                "is_poor": is_poor,
                "high_transmission": float(r["mal_prevalence"]) >= float(policy.transmission_threshold),
                "subsidy_rate": s_rate,
                "group_hh": group_hh,
                "avg_resource": avg_resource,
                "flows": flows,
                "flows0": flows0,
                "desired_new": desired_new,
                "baseline_new": baseline_new,
            })
    return records


def _priority_score(record: dict, policy: SubsidyPolicy, hs_per_case: float,
                    hh_per_case: float, daly_per_case: float) -> float:
    r = record["region_data"]
    if policy.allocation_rule == "poor_first":
        return (1.0 if record["is_poor"] else 0.0) * 1e6 + float(r["mal_prevalence"])
    if policy.allocation_rule == "high_transmission_first":
        return float(r["mal_prevalence"])
    if policy.allocation_rule == "cea_first":
        unit = max(record["flows"]["government_outlay"], EPS)
        # Approximate benefit per subsidized household using malaria risk and HH size.
        benefit = (float(r["household_size"]) * (float(r["mud"]) + float(r["bricks"])) *
                   float(r["mal_prevalence"]) * daly_per_case)
        fiscal_offset = (float(r["household_size"]) * (float(r["mud"]) + float(r["bricks"])) *
                         float(r["mal_prevalence"]) * (hs_per_case + hh_per_case))
        return (benefit + fiscal_offset) / unit
    return 0.0


def _voucher_fractions_by_year(
    records: List[dict],
    policy: SubsidyPolicy,
    hs_per_case: float,
    hh_per_case: float,
    daly_per_case: float,
    n_years: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return per-record, per-year voucher fractions.

    A fraction of desired subsidized adopters receive vouchers. With a binding
    cap, unvouchered households who would have adopted anyway are still allowed
    to adopt at full price in the flow calculation.
    """
    fractions = [np.zeros(n_years) for _ in records]
    required = np.zeros(n_years)
    shortfall = np.zeros(n_years)
    binding = np.zeros(n_years)

    for t in range(n_years):
        candidates = []
        for idx, rec in enumerate(records):
            unit_outlay = rec["flows"]["government_outlay"]
            desired = rec["desired_new"][t]
            if rec["subsidy_rate"] > 0 and unit_outlay > 0 and desired > 0:
                need = desired * unit_outlay
                candidates.append((idx, need))
                required[t] += need

        if not candidates:
            continue

        cap = policy.annual_budget_cap_usd
        if cap is None or required[t] <= cap:
            for idx, _need in candidates:
                fractions[idx][t] = 1.0
            continue

        cap = max(float(cap), 0.0)
        binding[t] = 1.0
        shortfall[t] = max(required[t] - cap, 0.0)

        if policy.allocation_rule == "proportional":
            f = cap / required[t] if required[t] > 0 else 0.0
            for idx, _need in candidates:
                fractions[idx][t] = f
            continue

        ranked = sorted(
            candidates,
            key=lambda x: _priority_score(records[x[0]], policy, hs_per_case, hh_per_case, daly_per_case),
            reverse=True,
        )
        remaining = cap
        for idx, need in ranked:
            if need <= 0:
                fractions[idx][t] = 0.0
                continue
            f = min(1.0, remaining / need)
            fractions[idx][t] = max(f, 0.0)
            remaining -= f * need
            if remaining <= 0:
                remaining = 0.0

    return fractions, required, shortfall, binding


def _empty_annual(n_years: int) -> Dict[str, np.ndarray]:
    keys = [
        "resource", "resource_undisc", "hh_oop", "hh_oop_undisc",
        "gov_outlay", "gov_outlay_undisc", "admin", "admin_undisc",
        "gov_transfer", "gov_transfer_undisc", "supplier_capture", "supplier_capture_undisc",
        "cases_averted", "cases_averted_disc", "adopters", "adopters_counterfactual",
        "additional_adopters", "subsidized_adopters", "poor_adopters", "poor_subsidized",
        "deadweight_transfer", "deadweight_transfer_undisc", "hs_savings", "hs_savings_undisc",
        "hh_savings", "hh_savings_undisc", "repairs", "functional_houses",
        "budget_required_undisc", "budget_cap_shortfall_undisc", "budget_cap_binding",
        "eligible_households", "poor_eligible_households",
    ]
    return {k: np.zeros(n_years) for k in keys}


def _simulate_policy_draw(
    params,
    regions: Dict[str, dict],
    policy: SubsidyPolicy,
    cost_mud: float,
    cost_brick: float,
    efficacy_sample: float,
    modifiable_samples: Dict[str, float],
    hs_per_case: float,
    hh_per_case: float,
    daly_per_case: float,
    adoption: AdoptionCalibration,
    reference_price_usd: float,
) -> Dict[str, np.ndarray]:
    years = params.years
    n = len(years)
    annual = _empty_annual(n)

    records = _make_group_records(
        params, regions, policy, cost_mud, cost_brick,
        modifiable_samples, adoption, reference_price_usd)

    voucher_frac, required, shortfall, binding = _voucher_fractions_by_year(
        records, policy, hs_per_case, hh_per_case, daly_per_case, n)

    annual["budget_required_undisc"] += required
    annual["budget_cap_shortfall_undisc"] += shortfall
    annual["budget_cap_binding"] += binding

    for idx, rec in enumerate(records):
        f = voucher_frac[idx]
        desired = rec["desired_new"]
        baseline = rec["baseline_new"]
        marginal = np.maximum(desired - baseline, 0.0)

        subsidized_new = f * desired
        actual_new = baseline + f * marginal
        unsubsidized_actual = np.maximum(actual_new - subsidized_new, 0.0)
        deadweight_houses = f * np.minimum(baseline, desired)

        functional, repairs = _functional_stock_and_repairs(actual_new, params)
        avg_resource = rec["avg_resource"]
        flows = rec["flows"]
        r = rec["region_data"]

        repair_resource = repairs * avg_resource * float(params.repair_cost_fraction)
        repair_gov_transfer = repair_resource * _safe_rate(policy.repair_subsidy_rate)
        repair_hh_oop = repair_resource - repair_gov_transfer

        protected_pop = (functional * float(r["household_size"]) *
                         (float(r["mud"]) + float(r["bricks"])))
        cases = protected_pop * float(r["mal_prevalence"]) * efficacy_sample

        for t in range(n):
            disc = 1.0 / ((1.0 + float(params.discount_rate)) ** t)

            new_resource = actual_new[t] * avg_resource
            resource = new_resource + repair_resource[t]
            gov_transfer = subsidized_new[t] * flows["government_transfer"] + repair_gov_transfer[t]
            admin = subsidized_new[t] * flows["admin"]
            gov_outlay = gov_transfer + admin
            hh_oop = (subsidized_new[t] * flows["household_price"] +
                      unsubsidized_actual[t] * avg_resource + repair_hh_oop[t])
            supplier_capture = subsidized_new[t] * flows["supplier_capture"]
            deadweight = deadweight_houses[t] * flows["government_transfer"]

            hs_save = cases[t] * hs_per_case
            hh_save = cases[t] * hh_per_case

            annual["resource_undisc"][t] += resource
            annual["resource"][t] += resource * disc
            annual["hh_oop_undisc"][t] += hh_oop
            annual["hh_oop"][t] += hh_oop * disc
            annual["gov_outlay_undisc"][t] += gov_outlay
            annual["gov_outlay"][t] += gov_outlay * disc
            annual["admin_undisc"][t] += admin
            annual["admin"][t] += admin * disc
            annual["gov_transfer_undisc"][t] += gov_transfer
            annual["gov_transfer"][t] += gov_transfer * disc
            annual["supplier_capture_undisc"][t] += supplier_capture
            annual["supplier_capture"][t] += supplier_capture * disc
            annual["deadweight_transfer_undisc"][t] += deadweight
            annual["deadweight_transfer"][t] += deadweight * disc
            annual["hs_savings_undisc"][t] += hs_save
            annual["hs_savings"][t] += hs_save * disc
            annual["hh_savings_undisc"][t] += hh_save
            annual["hh_savings"][t] += hh_save * disc
            annual["cases_averted"][t] += cases[t]
            annual["cases_averted_disc"][t] += cases[t] * disc
            annual["adopters"][t] += actual_new[t]
            annual["adopters_counterfactual"][t] += baseline[t]
            annual["additional_adopters"][t] += actual_new[t] - baseline[t]
            annual["subsidized_adopters"][t] += subsidized_new[t]
            annual["repairs"][t] += repairs[t]
            annual["functional_houses"][t] += functional[t]

            if rec["is_poor"]:
                annual["poor_adopters"][t] += actual_new[t]
                annual["poor_subsidized"][t] += subsidized_new[t]

        if rec["subsidy_rate"] > 0:
            annual["eligible_households"] += rec["group_hh"]
            if rec["is_poor"]:
                annual["poor_eligible_households"] += rec["group_hh"]

    return annual


# ---------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------
def run_policy_simulations(
    params,
    regions,
    policy: SubsidyPolicy,
    sim_inputs,
    n_simulations: int = 1000,
    seed0: int = 12345,
    adoption: Optional[AdoptionCalibration] = None,
):
    """Return per-draw national totals, with annual arrays included.

    Common random numbers are preserved because each policy uses the same
    seed0+i draw sequence.
    """
    adoption = adoption or AdoptionCalibration()
    ref_price = adoption.reference_price_usd
    if ref_price is None:
        ref_price = national_reference_price(params, regions)

    draws = []
    for i in range(int(n_simulations)):
        rng = np.random.default_rng(seed0 + i)
        cost_mud, cost_brick = draw_resource_costs(params, rng)
        efficacy_sample = _beta_draw(params.efficacy, params.efficacy_std, rng)
        hs_per_case, hh_per_case = draw_savings_per_case(sim_inputs, rng)
        daly_per_case = draw_daly_per_case(sim_inputs, rng)

        modifiable_samples = {
            name: float(np.clip(rng.normal(r["modifiable_homes"], 0.05), 0.0, 1.0))
            for name, r in regions.items()
        }

        annual = _simulate_policy_draw(
            params, regions, policy, cost_mud, cost_brick, efficacy_sample,
            modifiable_samples, hs_per_case, hh_per_case, daly_per_case,
            adoption, ref_price)

        total = {k: float(np.sum(v)) for k, v in annual.items()}
        total["dalys_averted"] = total["cases_averted_disc"] * daly_per_case
        total["dalys_averted_undisc"] = total["cases_averted"] * daly_per_case
        total["daly_per_case"] = daly_per_case
        total["reference_price_usd"] = float(ref_price)
        total["annual"] = {k: v.tolist() for k, v in annual.items()}
        draws.append(total)

    return draws


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------
def _mean(draws: Iterable[dict], key: str) -> float:
    return float(np.mean([d[key] for d in draws]))


def _pct(num: float, den: float) -> float:
    return 100.0 * num / den if abs(den) > EPS else np.nan


def budget_impact_row(draws, policy: SubsidyPolicy, sim_inputs=None):
    outlay_cash = _mean(draws, "gov_outlay_undisc")
    outlay_npv = _mean(draws, "gov_outlay")
    transfer_cash = _mean(draws, "gov_transfer_undisc")
    transfer_npv = _mean(draws, "gov_transfer")
    supplier_cash = _mean(draws, "supplier_capture_undisc")
    supplier_npv = _mean(draws, "supplier_capture")
    add = _mean(draws, "additional_adopters")
    dead = _mean(draws, "deadweight_transfer")
    hs_save = _mean(draws, "hs_savings")
    subsidized = _mean(draws, "subsidized_adopters")
    poor_subsidized = _mean(draws, "poor_subsidized")
    adopters = _mean(draws, "adopters")
    poor_adopters = _mean(draws, "poor_adopters")
    dalys = _mean(draws, "dalys_averted")
    resource = _mean(draws, "resource")
    admin_npv = _mean(draws, "admin")
    admin_cash = _mean(draws, "admin_undisc")

    return {
        "Policy": policy.name,
        "Households adopting": adopters,
        "Additional adopters vs no subsidy": add,
        "Subsidized households": subsidized,
        "Poor households reached": poor_adopters,
        "Poor share of adopters (%)": _pct(poor_adopters, adopters),
        "Poor share of subsidy recipients (%)": _pct(poor_subsidized, subsidized),
        "Government outlay, cash ($)": outlay_cash,
        "Government transfer, cash ($)": transfer_cash,
        "Admin cost, cash ($)": admin_cash,
        "Government outlay, NPV ($)": outlay_npv,
        "Government transfer, NPV ($)": transfer_npv,
        "Admin cost, NPV ($)": admin_npv,
        "Supplier capture/leakage, cash ($)": supplier_cash,
        "Supplier capture/leakage, NPV ($)": supplier_npv,
        "Health-system savings, NPV ($)": hs_save,
        "Net fiscal impact, NPV ($)": outlay_npv - hs_save,
        "Deadweight subsidy, NPV ($)": dead,
        "Deadweight share of transfers (%)": _pct(dead, transfer_npv),
        "Cost per additional adopter, cash ($)": outlay_cash / add if add > 0 else np.nan,
        "DALYs averted": dalys,
        "Fiscal cost per DALY, NPV ($)": (outlay_npv - hs_save) / dalys if dalys > 0 else np.nan,
        "Societal resource cost, NPV ($)": resource + admin_npv,
        "Budget cap binding years (%)": 100.0 * np.mean([np.mean(d["annual"]["budget_cap_binding"]) for d in draws]),
    }


def annual_budget_impact_rows(draws, policy: Optional[SubsidyPolicy] = None):
    if not draws:
        return []
    n = len(draws[0]["annual"]["gov_outlay_undisc"])
    rows = []
    for t in range(n):
        def amean(key):
            return float(np.mean([d["annual"][key][t] for d in draws]))

        outlay = amean("gov_outlay_undisc")
        hs = amean("hs_savings_undisc")
        rows.append({
            "Year": int(t),
            "Government outlay, cash ($)": outlay,
            "Government transfer, cash ($)": amean("gov_transfer_undisc"),
            "Admin/targeting cost, cash ($)": amean("admin_undisc"),
            "Supplier capture/leakage, cash ($)": amean("supplier_capture_undisc"),
            "Deadweight subsidy, cash ($)": amean("deadweight_transfer_undisc"),
            "Health-system savings, cash ($)": hs,
            "Net fiscal impact, cash ($)": outlay - hs,
            "Subsidized households": amean("subsidized_adopters"),
            "Additional adopters": amean("additional_adopters"),
            "Cases averted": amean("cases_averted"),
            "Budget required before cap ($)": amean("budget_required_undisc"),
            "Unfunded demand due to cap ($)": amean("budget_cap_shortfall_undisc"),
            "Probability cap binds": amean("budget_cap_binding"),
        })
    return rows


def _incremental_arrays(draws_policy, draws_comparator, perspective="societal"):
    dE, dC = [], []
    for p, c in zip(draws_policy, draws_comparator):
        d_daly = p["dalys_averted"] - c["dalys_averted"]
        d_hs = p["hs_savings"] - c["hs_savings"]
        d_hh_save = p["hh_savings"] - c["hh_savings"]

        if perspective == "government":
            cost = (p["gov_outlay"] - c["gov_outlay"]) - d_hs
        elif perspective == "program":
            cost = p["gov_outlay"] - c["gov_outlay"]
        elif perspective == "household":
            cost = (p["hh_oop"] - c["hh_oop"]) - d_hh_save
        elif perspective == "societal":
            d_res = (p["resource"] + p["admin"]) - (c["resource"] + c["admin"])
            cost = d_res - d_hs - d_hh_save
        else:
            raise ValueError("perspective must be societal, government, program, or household")

        dE.append(d_daly)
        dC.append(cost)
    return np.asarray(dE, dtype=float), np.asarray(dC, dtype=float)


def incremental_nmb_ceac(draws_policy, draws_comparator, sim_inputs, wtp_range,
                         perspective="societal"):
    """NMB-based incremental CEAC; handles negative costs safely."""
    dE, dC = _incremental_arrays(draws_policy, draws_comparator, perspective)
    ceac = [float(np.mean(lam * dE - dC > 0)) for lam in wtp_range]
    return ceac, dE, dC


def incremental_summary_row(policy_name, draws_policy, draws_comparator, threshold,
                            perspective="societal"):
    dE, dC = _incremental_arrays(draws_policy, draws_comparator, perspective)
    inmb = threshold * dE - dC
    return {
        "Policy": policy_name,
        "Delta DALYs": float(np.mean(dE)),
        "Delta cost, NPV ($)": float(np.mean(dC)),
        "Mean INMB ($)": float(np.mean(inmb)),
        "P(INMB > 0)": float(np.mean(inmb > 0)),
        "Incremental cost per DALY ($)": float(np.mean(dC) / np.mean(dE)) if np.mean(dE) > 0 else np.nan,
    }


def _total_cost_for_perspective(draw: dict, perspective: str = "societal") -> float:
    """Total discounted cost for one policy draw under a perspective."""
    if perspective == "government":
        return float(draw["gov_outlay"] - draw["hs_savings"])
    if perspective == "program":
        return float(draw["gov_outlay"])
    if perspective == "household":
        return float(draw["hh_oop"] - draw["hh_savings"])
    if perspective == "societal":
        return float(draw["resource"] + draw["admin"] - draw["hs_savings"] - draw["hh_savings"])
    raise ValueError("perspective must be societal, government, program, or household")


def policy_nmb_table(draws_by_policy: Dict[str, list], sim_inputs, perspective: str = "societal",
                     threshold: Optional[float] = None):
    """Expected NMB ranking and probability of being the best policy.

    This is useful because policy choice is a multi-option decision, not only a
    series of pairwise CEACs against one comparator.
    """
    lam = float(sim_inputs.threshold if threshold is None else threshold)
    names = list(draws_by_policy.keys())
    n = min(len(draws_by_policy[name]) for name in names)
    nmb = np.zeros((n, len(names)))

    for j, name in enumerate(names):
        for i, draw in enumerate(draws_by_policy[name][:n]):
            nmb[i, j] = lam * float(draw["dalys_averted"]) - _total_cost_for_perspective(draw, perspective)

    best_idx = np.argmax(nmb, axis=1)
    rows = []
    for j, name in enumerate(names):
        draws = draws_by_policy[name][:n]
        costs = [_total_cost_for_perspective(d, perspective) for d in draws]
        dalys = [d["dalys_averted"] for d in draws]
        rows.append({
            "Policy": name,
            "Mean DALYs": float(np.mean(dalys)),
            "Mean cost, NPV ($)": float(np.mean(costs)),
            "Expected NMB ($)": float(np.mean(nmb[:, j])),
            "NMB lower 95% ($)": float(np.percentile(nmb[:, j], 2.5)),
            "NMB upper 95% ($)": float(np.percentile(nmb[:, j], 97.5)),
            "Probability best (%)": 100.0 * float(np.mean(best_idx == j)),
        })

    rows = sorted(rows, key=lambda r: r["Expected NMB ($)"], reverse=True)
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("Policy")
    except Exception:
        return rows


def default_scenarios(
    poverty_var: str = "mdp",
    transmission_threshold: float = 0.10,
    pass_through: float = 1.0,
    admin_cost_per_hh_usd: float = 0.0,
    annual_budget_cap_usd: Optional[float] = None,
    allocation_rule: str = "proportional",
):
    pov_label = poverty_var.upper() if poverty_var == "mdp" else poverty_var.title()

    def pol(name, rate, target):
        return SubsidyPolicy(
            name=name,
            rate=rate,
            target=target,
            poverty_var=poverty_var,
            transmission_threshold=transmission_threshold,
            pass_through=pass_through,
            admin_cost_per_hh_usd=admin_cost_per_hh_usd,
            annual_budget_cap_usd=annual_budget_cap_usd,
            allocation_rule=allocation_rule,
        )

    return [
        pol("No public subsidy", 0.0, "universal"),
        pol("Universal 25%", 0.25, "universal"),
        pol("Universal 50%", 0.50, "universal"),
        pol("Universal 75%", 0.75, "universal"),
        pol("Universal 100%", 1.00, "universal"),
        pol(f"Poverty-targeted 50% ({pov_label})", 0.50, "poor"),
        pol(f"Poverty-targeted 100% ({pov_label})", 1.00, "poor"),
        pol("High-transmission 100%", 1.00, "high_transmission"),
        pol(f"Hybrid 100%: poor OR high-transmission ({pov_label})", 1.00, "hybrid"),
        pol(f"Poor in high-transmission 100% ({pov_label})", 1.00, "hybrid_and"),
    ]


# ---------------------------------------------------------------------
# Decision-visualization helpers used by the Streamlit subsidy tab
# ---------------------------------------------------------------------
def regional_targeting_rows(
    regions: Dict[str, dict],
    poverty_var: str = "mdp",
    transmission_threshold: float = 0.10,
):
    """Region-level targeting inputs for the priority bubble plot.

    Returns plain dictionaries so app.py can turn them into a DataFrame
    without adding pandas as a policy.py dependency.
    """
    raw = []
    for name, r in regions.items():
        poverty = float(r.get(poverty_var, 0.0))
        prevalence = float(r.get("mal_prevalence", 0.0))
        households = float(r.get("households", 0.0))
        modifiable = float(r.get("modifiable_homes", 0.0))
        modifiable_hh = households * modifiable
        rdt = float(r.get("rdt_pos_24", np.nan))
        priority = prevalence * max(poverty, 0.0) * modifiable_hh
        raw.append({
            "Region": name,
            "Malaria prevalence (%)": 100.0 * prevalence,
            "Poverty share (%)": 100.0 * poverty,
            "Modifiable households": modifiable_hh,
            "2024 RDT-positive cases": rdt,
            "High transmission": prevalence >= float(transmission_threshold),
            "Targeting priority index": priority,
        })

    if not raw:
        return []

    median_poverty = float(np.median([row["Poverty share (%)"] for row in raw]))
    median_prev = float(np.median([row["Malaria prevalence (%)"] for row in raw]))
    for row in raw:
        high_pov = row["Poverty share (%)"] >= median_poverty
        high_tx = row["Malaria prevalence (%)"] >= 100.0 * float(transmission_threshold)
        if high_pov and high_tx:
            quadrant = "High poverty / high transmission"
        elif high_tx:
            quadrant = "High transmission"
        elif high_pov:
            quadrant = "High poverty"
        else:
            quadrant = "Lower priority by these two criteria"
        row["Priority quadrant"] = quadrant
        row["Median poverty share (%)"] = median_poverty
        row["Median malaria prevalence (%)"] = median_prev
    return raw


def _row_value(row, key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, default)
    except AttributeError:
        val = default
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = default
    if np.isnan(val):
        return float(default)
    return val


def fiscal_decomposition_from_budget_row(row):
    """Mutually exclusive NPV public-outlay components for the waterfall.

    The budget-impact row reports deadweight transfer and supplier capture.
    Those concepts can overlap, so this function decomposes the government
    transfer into mutually exclusive categories:
      - passed-through subsidy for marginal additional adopters
      - passed-through subsidy for likely inframarginal adopters
      - supplier capture/leakage
      - admin/targeting cost
    """
    outlay = max(_row_value(row, "Government outlay, NPV ($)"), 0.0)
    transfer = max(_row_value(row, "Government transfer, NPV ($)", outlay), 0.0)
    admin = max(_row_value(row, "Admin cost, NPV ($)"), 0.0)
    supplier = max(_row_value(row, "Supplier capture/leakage, NPV ($)"), 0.0)
    deadweight_transfer = min(max(_row_value(row, "Deadweight subsidy, NPV ($)"), 0.0), transfer)

    capture_share = supplier / transfer if transfer > EPS else 0.0
    capture_share = min(max(capture_share, 0.0), 1.0)
    marginal_transfer = max(transfer - deadweight_transfer, 0.0)

    productive_passed = marginal_transfer * (1.0 - capture_share)
    deadweight_passed = deadweight_transfer * (1.0 - capture_share)

    return [
        {
            "Component": "Passed-through subsidy to marginal adopters",
            "Value ($)": productive_passed,
            "Interpretation": "Public transfer that lowers price for households induced to adopt.",
        },
        {
            "Component": "Passed-through deadweight subsidy",
            "Value ($)": deadweight_passed,
            "Interpretation": "Transfer reaching households likely to adopt even without subsidy.",
        },
        {
            "Component": "Supplier capture/leakage",
            "Value ($)": supplier,
            "Interpretation": "Public transfer not passed through as a lower household price.",
        },
        {
            "Component": "Admin/targeting cost",
            "Value ($)": admin,
            "Interpretation": "Real implementation cost of delivering or verifying subsidy.",
        },
    ]
