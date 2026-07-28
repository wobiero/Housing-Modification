# =====================================================================
# FILE: simulation.py
# Coverage scale-up, the Monte-Carlo engine, and the stats dataframe.
# =====================================================================
import numpy as np
import pandas as pd
import streamlit as st
from itertools import accumulate
from scipy.stats import gamma, beta
from scipy.special import expit

from data import human_cap
from stats import (gamma_stats, beta_stats, summarize, cumulate,
                   scale_cases, add_metric)


def sigmoid_scaleup(t, max_coverage, growth_rate, midpoint_year, params, region_data, shape=.4):
    """CHANGED (modularization): region_data is now explicit (was a global).
    The adoption probability is driven by the passed region_data's wall
    proportions and by params' costs/subsidy."""
    price_ugx = (params.cost_mud["mean"] * region_data["mud"] +
                 params.cost_brick["mean"] * region_data["bricks"]) * 3594
    price_ugx *= (1 - params.subsidy)
    prob_adopt = expit(134.37329 - .000336 * price_ugx)
    return max_coverage * prob_adopt / ((1 + np.exp(-growth_rate * (t - midpoint_year))) ** shape)


def effective_coverage(years, max_coverage, growth_rate, midpoint_year, params, region_data):
    n_years = len(years)
    coverage = sigmoid_scaleup(years, max_coverage, growth_rate, midpoint_year,
                               params, region_data, shape=0.4)
    effective = np.zeros(n_years)
    for t in range(n_years):
        for k in range(t + 1):
            new_houses = coverage[k] - coverage[k - 1] if k > 0 else coverage[0]
            survival = (1 - params.failure_rate) ** (t - k)
            effective[t] += new_houses * survival
    return effective


def _functional_stock_and_repairs(new_houses, failure_rate, annual_repair_rate):
    """Track functioning modified houses, annual repairs, and the outstanding
    (unrepaired) failed stock.

    This mirrors policy._functional_stock_and_repairs so the regional cost
    engine and the subsidy engine share one repair model:
      - A share (failure_rate) of the *currently functioning* stock fails each
        year.
      - A share (annual_repair_rate) of the *accumulated failed* stock is
        repaired each year and RETURNS to the functioning stock.
      - Repair cost is charged on repairs performed that year, NOT on the
        cumulative probability of ever having failed (which double-counted
        the same houses every subsequent year in the previous version).

    Returns
    -------
    functional        : houses providing protection each year (repaired houses
                        count again, per the requested behaviour)
    repairs           : houses repaired each year (repair cost is charged here)
    failed_outstanding: failed stock still awaiting repair at year-end
    """
    n = len(new_houses)
    functional = np.zeros(n)
    repairs = np.zeros(n)
    failed_outstanding = np.zeros(n)
    failed_stock = 0.0

    for t in range(n):
        previous_functional = functional[t - 1] if t > 0 else 0.0
        failed_this_year = previous_functional * float(failure_rate)
        repaired_this_year = failed_stock * float(annual_repair_rate)
        failed_stock = max(failed_stock + failed_this_year - repaired_this_year, 0.0)
        functional[t] = max(previous_functional - failed_this_year
                            + repaired_this_year + new_houses[t], 0.0)
        repairs[t] = repaired_this_year
        failed_outstanding[t] = failed_stock

    return functional, repairs, failed_outstanding


def run_simulations(params, regions, n_simulations=100):
    """Coverage is evaluated PER REGION using each region's own per-house cost,
    so the WTP-based adoption probability reflects that region's wall mix.

    Repair accounting (CHANGED): houses that fail are drawn from the
    functioning stock at params.failure_rate; a share (params.annual_repair_rate)
    of the failed stock is repaired each year and becomes protective again.
    Repair cost is billed only on repairs performed in that year. The previous
    version charged repair cost on the whole cumulative ever-failed pool every
    year (a ~7x over-count over a 10-year horizon) and never let a repaired
    house avert cases again.
    """
    all_results = []

    mu = params.efficacy
    sigma = params.efficacy_std
    efficacy_alpha = ((1 - mu) / sigma ** 2 - 1 / mu) * mu ** 2
    efficacy_beta = efficacy_alpha * (1 / mu - 1)

    failure_rate = params.failure_rate
    annual_repair_rate = params.annual_repair_rate
    repair_cost_fraction = params.repair_cost_fraction

    for sim_idx in range(n_simulations):
        np.random.seed(12345 + sim_idx)
        # Resource cost is drawn at the FULL mean -- a subsidy is a transfer,
        # not a reduction in real materials/labour cost. The subsidy affects
        # (a) the household-facing price in sigmoid_scaleup (adoption) and
        # (b) the government outlay stream added below. It must NOT shrink the
        # societal resource cost.
        cost_mud = gamma.rvs(a=params.cost_mud["mean"] ** 2 / params.cost_mud["std"] ** 2,
                             scale=params.cost_mud["std"] ** 2 / params.cost_mud["mean"])
        cost_brick = gamma.rvs(a=params.cost_brick["mean"] ** 2 / params.cost_brick["std"] ** 2,
                               scale=params.cost_brick["std"] ** 2 / params.cost_brick["mean"])
        efficacy_sample = beta.rvs(efficacy_alpha, efficacy_beta)

        region_results = []
        for region_name, region_data in regions.items():
            # Per-region WTP coverage from THIS region's wall mix / price.
            coverage_all_years = sigmoid_scaleup(
                params.years, params.max_coverage, params.growth_rate,
                params.midpoint_year, params, region_data, params.shape)

            modifiable_sample = np.clip(
                np.random.normal(loc=region_data["modifiable_homes"], scale=0.05), 0, 1)
            modifiable_hh = region_data["households"] * modifiable_sample
            avg_cost = cost_mud * region_data["mud"] + cost_brick * region_data["bricks"]

            houses_modified = np.diff(coverage_all_years, prepend=0) * modifiable_hh
            houses_modified = np.maximum(houses_modified, 0.0)
            cumulative_houses = np.cumsum(houses_modified)

            # Stock-flow: functioning houses, repairs, and outstanding failures.
            functional, repairs, failed_outstanding = _functional_stock_and_repairs(
                houses_modified, failure_rate, annual_repair_rate)

            n_years = len(params.years)
            repair_costs = np.zeros(n_years)
            costs = np.zeros(n_years)
            undiscounted_costs = np.zeros(n_years)
            cases_averted = np.zeros(n_years)
            effective_fraction = np.zeros(n_years)

            wall_mod_share = region_data["mud"] + region_data["bricks"]

            for t, year in enumerate(params.years):
                discount_factor = 1 / ((1 + params.discount_rate) ** t)

                # Repair cost is billed on repairs performed this year only.
                repair_costs[t] = repairs[t] * avg_cost * repair_cost_fraction

                new_construction = houses_modified[t] * avg_cost
                undiscounted_costs[t] = new_construction + repair_costs[t]
                costs[t] = undiscounted_costs[t] * discount_factor

                # Protection comes from the CURRENT functioning stock (repaired
                # houses are protective again).
                effective_fraction[t] = (functional[t] / modifiable_hh
                                         if modifiable_hh > 0 else 0.0)
                effective_fraction[t] = np.clip(effective_fraction[t], 0,
                                                coverage_all_years[t])

                protected_pop = (functional[t] * wall_mod_share
                                 * region_data["household_size"])
                cases_averted[t] = (protected_pop * region_data["mal_prevalence"]
                                    * efficacy_sample)

            region_results.append({
                "Region": region_name,
                "Cost_Per_HH": float(avg_cost),
                "Cases_Averted": cases_averted.tolist(),
                "New_Construction_Costs": (houses_modified * avg_cost).tolist(),
                # Government pays the subsidy share of the resource cost
                # (undiscounted cash outlay). The resource cost itself is
                # unchanged -- this is the transfer, tracked separately.
                "Government_Subsidy_Outlay": (houses_modified * avg_cost * params.subsidy).tolist(),
                "Repair_Costs": repair_costs.tolist(),
                "Total_Costs": costs.tolist(),
                "Undiscounted_Costs": undiscounted_costs.tolist(),
                "Coverage_Target": coverage_all_years.tolist(),
                "Houses_Modified_Annual": houses_modified.tolist(),
                "Houses_Modified_Cumulative": cumulative_houses.tolist(),
                # Now the outstanding (unrepaired) failed stock per year, so
                # analyze_results' Repair_Percentage = currently-broken share.
                "Houses_Needing_Repair": failed_outstanding.tolist(),
                "Repairs_Performed_Annual": repairs.tolist(),
                "Functional_Houses": functional.tolist(),
                "Effective_Coverage": (effective_fraction * 100).tolist(),
            })
        all_results.append(region_results)
    return all_results


@st.cache_data
def analyze_results(results):
    """Flatten and summarize simulation results (writes a preview to the UI)."""
    flat_results = [region for sim in results for region in sim]
    df = pd.DataFrame(flat_results)

    summary = df.groupby('Region').agg({
        'Cost_Per_HH': ['mean', 'std'],
        'Cases_Averted': lambda x: np.mean([val[-1] for val in x]),
        'Total_Costs': lambda x: np.mean([sum(val) for val in x]),
        'Houses_Modified_Cumulative': lambda x: np.mean([val[-1] for val in x]),
        'Houses_Needing_Repair': lambda x: np.mean([val[-1] for val in x]),
        'Repair_Costs': lambda x: np.mean([sum(val) for val in x]),
    })
    summary.columns = ['Cost_Per_HH_Mean', 'Cost_Per_HH_Std', 'Cases_Averted',
                       'Total_Cost', 'Houses_Modified', 'Houses_Needing_Repair',
                       'Total_Repair_Cost']

    st.write("\nCumulative Regional Summary (Year 10):")
    st.write(summary.round(2))
    st.write(summary['Cases_Averted'].sum())
    st.write(summary['Houses_Modified'].sum())

    if 'Houses_Needing_Repair' in df.columns and 'Houses_Modified_Cumulative' in df.columns:
        summary['Repair_Percentage'] = (summary['Houses_Needing_Repair'] /
                                        summary['Houses_Modified'] * 100).round(1)
    return df


def create_stats_dataframe(nested_list, sim):
    """nested_list: list of sims, each a list of 10 yearly averted-case values.
    sim: SimulationInputs to use (pass updated_sim_data to honour the sliders)."""
    by_year = [[s[i] for s in nested_list] for i in range(10)]

    sev_mal = scale_cases(by_year, lambda: beta_stats(sim.severe_mean, sim.severe_sd, 1)[0])
    cer_mal = scale_cases(by_year, lambda: beta_stats(sim.cerebral_malaria_mean, sim.cerebral_malaria_sd, 1)[0])
    cer_an = scale_cases(by_year, lambda: beta_stats(sim.cerebral_anemia, sim.cerebral_anemia_sd, 1)[0])
    sev_anemia = scale_cases(by_year, lambda: beta_stats(sim.severe_anemia_mean, sim.severe_anemia_sd, 1)[0])
    deaths = scale_cases(by_year, lambda: beta_stats(sim.deaths, sim.deaths_sd, 1)[0])

    vsl = [[d * sim.disc_lifespan * gamma_stats(sim.mean_vsly, sim.std_vsly, 1)[0]
            for d in yr] for yr in deaths]
    human_cap_samples = [[d * human_cap for d in yr] for yr in deaths]
    opd = scale_cases(by_year, lambda: gamma_stats(sim.opd_cost, sim.opd_cost_sd, 1)[0])
    ipd = scale_cases(by_year, lambda: gamma_stats(sim.ipd_cost, sim.ipd_cost_sd, 1)[0])
    um_hh = scale_cases(by_year, lambda: gamma_stats(sim.hh_um_cost, sim.hh_um_cost_sd, 1)[0])
    sm_hh = scale_cases(sev_mal, lambda: gamma_stats(sim.hh_sm_cost, sim.hh_sm_cost_sd, 1)[0])

    data = {}
    data['averted_cases'] = [np.mean(x) for x in by_year]
    data['averted_std_dev'] = [np.std(x) for x in by_year]
    data['averted_cases_min'] = [np.min(x) for x in by_year]
    data['averted_cases_max'] = [np.max(x) for x in by_year]
    data['cumulative_averted_cases'] = list(accumulate(data['averted_cases']))
    data['cumulative_averted_cases_lower'] = list(accumulate(data['averted_cases_min']))
    data['cumulative_averted_cases_upper'] = list(accumulate(data['averted_cases_max']))

    add_metric(data, 'severe_malaria', sev_mal)
    add_metric(data, 'severe_anemia', sev_anemia)
    add_metric(data, 'cerebral_malaria', cer_mal)
    add_metric(data, 'cerebral_anemia', cer_an)
    add_metric(data, 'deaths', deaths, cum_name='cum_deaths')  # preserves odd cum naming
    add_metric(data, 'opd_cost', opd)
    add_metric(data, 'ipd_cost', ipd)
    add_metric(data, 'um_hh_cost', um_hh)
    add_metric(data, 'sm_hh_cost', sm_hh)
    add_metric(data, 'vsl', vsl)
    add_metric(data, 'death_human_cap', human_cap_samples)

    return pd.DataFrame(data)