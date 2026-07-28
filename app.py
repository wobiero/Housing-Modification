
# =====================================================================
# Author: Walter Ochieng'
# Organization: CDC/GHC/OD
# =====================================================================
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import asdict

from data import regions, param_labels, terms_conditions
from config import SimulationInputs, Parameters
from simulation import (run_simulations, analyze_results,
                        create_stats_dataframe, effective_coverage)
from economics import compute_ceac, add_savings_columns
from plotting import (download_fig, dollar_formatter,
                      plot_ce_plane, plot_tornado_diagram)
from sensitivity import tornado_analysis
from policy import (run_policy_simulations, budget_impact_row, incremental_nmb_ceac,
                    default_scenarios, SubsidyPolicy, AdoptionCalibration,
                    annual_budget_impact_rows, incremental_summary_row,
                    national_reference_price, policy_nmb_table,
                    regional_targeting_rows, fiscal_decomposition_from_budget_row)
st.set_page_config(page_title="Housing Modification Economic Analysis", layout="wide")
with st.spinner("Loading Housing Modification Economic Analysis Tool ....."):
    time.sleep(1)

st.title("🏠 Uganda Housing Modification for Malaria Economic Analysis")

tabs = st.tabs(["About", "Summary Inputs", "Regional Summary",
                "National Summary and Notes", "Subsidy Policy", "Contact Us"])

with tabs[0]:
    with st.expander("Terms and conditions"):
        st.write(terms_conditions)

# ----- Sidebar: region + region parameter sliders -----
region = st.sidebar.selectbox("Select a region:", sorted(list(regions.keys())))
ug_regions = list(regions.keys())

with st.sidebar.expander(f"Adjust Parameters for {region}"):
    region_data = regions[region]
    updated_region_data = {}
    for key, value in region_data.items():
        label = param_labels.get(key, key)
        if isinstance(value, int):
            updated_value = st.slider(label, int(value * .5), int(value * 1.5),
                                      value, step=1, format="%d")
        elif isinstance(value, float):
            updated_value = st.slider(label, 0.0, (value * 1.5),
                                      float(value), step=.01, format="%.3f")
        else:
            updated_value = value
        updated_region_data[key] = updated_value

adj_annual_failure_rate = (updated_region_data["mud"] * .1) + (updated_region_data["bricks"] * .05)

# ----- Sidebar: global model parameters -----
sim_data = SimulationInputs()
sim_data_dict = asdict(sim_data)

with st.sidebar.expander("Global Model Parameters"):
    updated_data = {}
    for field_name, value in sim_data_dict.items():
        label = field_name.replace("_", " ").upper()
        if isinstance(value, float):
            if 0 <= value <= 1:
                slider_value = st.slider(label, 0.0, 1.0, float(value), step=.01, format="%.3f")
            else:
                slider_value = st.slider(label, float(value * .5), float(value * 1.5),
                                         float(value), step=float(value * .05), format="%.3f")
        elif isinstance(value, int):
            slider_value = st.slider(label, int(value * .5), int(value * 1.5), value, step=1)
        else:
            slider_value = value
        updated_data[field_name] = slider_value

updated_sim_data = SimulationInputs(**updated_data)

# ----- Calibration parameters -----
params = Parameters(annual_repair_rate=adj_annual_failure_rate)
params.update_from_sliders()

# =====================================================================
# Tab 1: Summary inputs
# =====================================================================
with tabs[1]:
    np.random.seed(42)
    with st.expander("Cumulative Summary Values for Year 10"):
        results = run_simulations(params, regions, n_simulations=100)
        final_df = analyze_results(results)

    cases_averted = final_df["Cases_Averted"][final_df["Region"] == region].tolist()
    stats_df = create_stats_dataframe(cases_averted, updated_sim_data)

    y = final_df["Total_Costs"][final_df["Region"] == region].tolist()
    z = final_df["Undiscounted_Costs"][final_df["Region"] == region].tolist()
    v = final_df['Houses_Modified_Annual'][final_df['Region'] == region].tolist()
    w = final_df['Houses_Modified_Cumulative'][final_df['Region'] == region].tolist()

    stats_df["total_house_costs"] = [sum(item) / len(item) for item in zip(*y)]
    stats_df["undiscounted_house_costs"] = [sum(item) / len(item) for item in zip(*z)]
    stats_df["house_num_mod_annual"] = [sum(item) / len(item) for item in zip(*v)]
    stats_df["house_num_mod_cum"] = [sum(item) / len(item) for item in zip(*w)]

    with st.expander(f"View Updated Region Data for {region}:"):
        labeled_data = {param_labels.get(k, k): val for k, val in updated_region_data.items()}
        st.dataframe(pd.DataFrame.from_dict(labeled_data, orient="index", columns=["Value"]))

    housing_narrative = f"""The 2024 Uganda National Census estimates the population of {region} to be {updated_region_data["population"]:,.0f}, with {updated_region_data["pop_u5"]:,.0f}
    being children under-five.
    The estimated population growth rate was {updated_region_data["pop_growth_rate"] * 100}%.
    The number of households according to the census were {updated_region_data["households"]:,.0f}.
    The reported positive malaria RDTs in {region} from the DHIS-2 in 2024 was {updated_region_data["rdt_pos_24"]:,.0f}. Approximately {updated_region_data["bricks"] * 100:,.1f}% houses were made of brick
    while {updated_region_data["mud"] * 100:,.1f}% were mud houses. The estimated proportion of houses that could be modified was {updated_region_data["modifiable_homes"] * 100:,.1f}%.
    {updated_region_data["subsistence"] * 100:,.1f}% were engaged in subsistence agriculture.
    """
    with st.expander(f"Regional Summary for {region}"):
        st.write(housing_narrative)

# =====================================================================
# Tab 2: Regional summary
# =====================================================================
with tabs[2]:
    with st.expander(f"Summary Data for {region}"):
        stats_df.index.name = "Year"
        mean_screening_costs = final_df["Cost_Per_HH"][final_df["Region"] == region].mean()
        stats_df["cum_screening_costs"] = stats_df["house_num_mod_cum"] * mean_screening_costs

        add_savings_columns(stats_df, updated_sim_data)
        stats_df["total_house_costs_cum"] = stats_df["total_house_costs"].cumsum()
        st.dataframe(stats_df.style.format("{:,.2f}"))

        st.write(f"The discounted costs of modifying all houses in {region} over 10 years is \\${stats_df['total_house_costs'].sum():,.2f}")
        st.write(f"The undiscounted costs of modifying all houses in {region} over 10 years is \\${stats_df['undiscounted_house_costs'].sum():,.2f}")

    with st.expander(f"Net Savings Societal Perspective: {region}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        net = stats_df["societal_savings_cum_death"] - stats_df["total_house_costs_cum"]
        net_lo = stats_df["societal_savings_cum_death_lower"] - stats_df["total_house_costs_cum"]
        net_hi = stats_df["societal_savings_cum_death_upper"] - stats_df["total_house_costs_cum"]
        plt.plot(stats_df.index, net, label="Mean Costs")
        plt.plot(stats_df.index, net_lo, label="Lower Costs")
        plt.plot(stats_df.index, net_hi, label="Upper Costs")
        plt.fill_between(stats_df.index, net_lo, net_hi, color="g", alpha=.2, zorder=-2)
        plt.xlabel("Time in Years")
        plt.ylabel("Dollars ($)")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.title(f"Trends in estimated cumulative cost savings: {region}")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        plt.legend(loc="best")
        plt.grid(axis="both", alpha=.2)
        st.pyplot(fig, width="stretch")
        download_fig(fig, f"Saving_Trends_{region}.png", "Download PNG")

    with st.expander(f"Annual Economic Savings from Averted Deaths: {region}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(stats_df.index, stats_df["vsl"], label="Mean")
        plt.fill_between(stats_df.index, stats_df["vsl_lower"], stats_df["vsl_upper"],
                         color='g', alpha=0.1, zorder=-2)
        plt.grid(axis="both", alpha=.2)
        plt.xlabel("Time in Years")
        plt.ylabel("Economic Costs ($)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.title(f"Annual Economic Savings - Deaths Averted \n{region}", fontsize=18)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        st.pyplot(fig)
        download_fig(fig, f"Annual_Savings_{region}.png", "Download PNG")

    with st.expander(f"Cumulative Savings from Deaths Averted: {region}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(stats_df.index, stats_df["vsl_cum"], label="Mean")
        plt.grid(axis="both", alpha=.2)
        plt.fill_between(stats_df.index, stats_df["vsl_cum_lower"], stats_df["vsl_cum_upper"],
                         color='g', alpha=0.1, zorder=-2)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.title(f"Cumulative Economic savings - Deaths Averted \n{region}", fontsize=18)
        st.pyplot(fig)
        download_fig(fig, f"Cum_Savings_{region}.png", "Download PNG")

    with st.expander(f"Disability Adjusted Life Years Averted: {region}"):
        s = updated_sim_data
        N_DALY = 10_000
        rng_daly = np.random.default_rng(2024)

        def _gamma_draws(mean, sd, size=N_DALY):
            """Vectorised twin of policy._gamma_draw (mean/sd parameterisation)."""
            if mean <= 0 or sd <= 0:
                return np.full(size, float(mean))
            shape = (mean / sd) ** 2
            scale = (sd ** 2) / mean
            return rng_daly.gamma(shape, scale, size=size)

        def _beta_draws(mean, sd, size=N_DALY):
            """Vectorised twin of policy._beta_draw (mean/sd parameterisation)."""
            eps = 1e-6
            mean = min(max(float(mean), eps), 1 - eps)
            max_sd = np.sqrt(mean * (1 - mean))
            sd = min(max(float(sd), 1e-8), max_sd - 1e-8)
            var = sd ** 2
            nu = mean * (1 - mean) / var - 1.0
            return rng_daly.beta(mean * nu, (1 - mean) * nu, size=size)

        def _band(mult):
            """(mean, 2.5th, 97.5th) of a DALY-multiplier draw array."""
            return (float(np.mean(mult)),
                    float(np.percentile(mult, 2.5)),
                    float(np.percentile(mult, 97.5)))

        # Per-case morbidity multipliers = duration * disability weight / 365.
        # (Disease shares are already baked into the stats_df counts, so we do
        #  NOT re-multiply by severe/cerebral probabilities here.)
        multipliers = {
            "uncomp":          (_gamma_draws(s.duration_um, s.duration_um_sd)
                                * _beta_draws(s.dw_um, s.dw_um_sd) / 365.0),
            "severe_mal":      (_gamma_draws(s.duration_sm, s.duration_sm_sd)
                                * _beta_draws(s.dw_sm, s.dw_sm_sd) / 365.0),
            "severe_anemia":   (_gamma_draws(s.duration_severe_anemia, s.duration_severe_anemia_sd)
                                * _beta_draws(s.dw_sev_anemia, s.dw_sev_anemia_sd) / 365.0),
            "cerebral":        (_gamma_draws(s.duration_cerebral, s.duration_cerebral_sd)
                                * _beta_draws(s.dw_cerebral, s.dw_cerebral_sd) / 365.0),
            # cerebral-anemia reuses the cerebral duration (as in the original)
            "cerebral_anemia": (_gamma_draws(s.duration_cerebral, s.duration_cerebral_sd)
                                * _beta_draws(s.dw_cerebral_anemia, s.dw_cerebral_anemia_sd) / 365.0),
        }
        count_cols = {
            "uncomp": "averted_cases",
            "severe_mal": "severe_malaria",
            "severe_anemia": "severe_anemia",
            "cerebral": "cerebral_malaria",
            "cerebral_anemia": "cerebral_anemia",
        }

        qaly_df = pd.DataFrame(index=stats_df.index)

        # Mortality DALYs: disc_lifespan is a fixed input, so bands come from the
        # death-count CI (deaths_lower / deaths_upper are 2.5/97.5 percentiles).
        qaly_df["death_daly_mean"]  = stats_df["deaths"]       * s.disc_lifespan
        qaly_df["death_daly_lower"] = stats_df["deaths_lower"] * s.disc_lifespan
        qaly_df["death_daly_upper"] = stats_df["deaths_upper"] * s.disc_lifespan

        # Morbidity DALYs: expected case count * multiplier CI. Because counts are
        # non-negative, the multiplier percentiles scale linearly year by year.
        for name, mult in multipliers.items():
            m, lo, hi = _band(mult)
            counts = stats_df[count_cols[name]].to_numpy()
            qaly_df[f"{name}_daly_mean"]  = counts * m
            qaly_df[f"{name}_daly_lower"] = counts * lo
            qaly_df[f"{name}_daly_upper"] = counts * hi

        for band in ["mean", "lower", "upper"]:
            qaly_df[f"total_daly_averted_{band}"] = (
                qaly_df[f"cerebral_daly_{band}"] + qaly_df[f"uncomp_daly_{band}"] +
                qaly_df[f"severe_anemia_daly_{band}"] + qaly_df[f"cerebral_anemia_daly_{band}"] +
                qaly_df[f"death_daly_{band}"] + qaly_df[f"severe_mal_daly_{band}"])
            qaly_df[f"total_daly_averted_cum_{band}"] = qaly_df[f"total_daly_averted_{band}"].cumsum()

        st.dataframe(qaly_df.style.format("{:,.1f}"))
        st.caption(
            "Morbidity DALY bands (2.5–97.5%) reflect Monte-Carlo uncertainty in "
            "episode duration and disability weights at expected case counts. "
            "Mortality DALY bands reflect the case-count credible interval "
            "(discounted life expectancy is treated as fixed)."
        )

        averted_ip_costs = stats_df[["ipd_cost", "ipd_cost_lower", "ipd_cost_upper",
                                     "ipd_cost_cum", "ipd_cost_cum_lower", "ipd_cost_cum_upper"]][:]

    with st.expander(f"Averted Inpatient Costs for {region}"):
        averted_ip_costs.index.name = "Year"
        st.dataframe(averted_ip_costs.style.format("${:,.2f}"))
        ipd_cost_cum = averted_ip_costs["ipd_cost_cum"][-1:].values[0]
        ipd_cost_min = averted_ip_costs["ipd_cost_cum_lower"][-1:].values[0]
        ipd_cost_max = averted_ip_costs["ipd_cost_cum_upper"][-1:].values[0]
        st.write(f"Cumulative averted inpatient costs in 10 years is \\${ipd_cost_cum:,.0f} [\\${ipd_cost_min:,.0f}: \\${ipd_cost_max:,.0f}]")

    with st.expander(f"Severe Malaria Anemia Averted Costs for {region}"):
        st.write("32.8% of severe malaria anemia patients get blood transfusion. [Ackerman et. al (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611367/)")
        sma_df = stats_df[["severe_anemia", "severe_anemia_lower", "severe_anemia_upper",
                           "severe_anemia_cum", "severe_anemia_cum_lower", "severe_anemia_cum_upper"]].copy()
        for col in list(sma_df.columns):
            sma_df[f"{col}_costs"] = sma_df[col] * updated_sim_data.transfusion_cost
        st.dataframe(sma_df.style.format("{:,.2f}"))
        transfusion_costs_mean = sma_df["severe_anemia_cum_costs"][-1:].values[0]
        transfusion_costs_lower = sma_df["severe_anemia_cum_lower_costs"][-1:].values[0]
        transfusion_costs_upper = sma_df["severe_anemia_cum_upper_costs"][-1:].values[0]
        st.write(f"""The averted cumulative costs of treating severe malaria anemia in {region} in 10 years is \\${transfusion_costs_mean:,.0f}
                  [\\${transfusion_costs_lower:,.0f}: \\${transfusion_costs_upper:,.0f}]. This assumes approximately {sma_df["severe_anemia_cum"][-1:].values[0]:,.0f}
                  severe malaria anemia cases will be averted in that timeframe. We assume that {updated_sim_data.prop_sam_transfused * 100:,.1f}% of the SMA cases will receive blood transfusions.
                  The average cost of a blood transfusion is estimated at \\${updated_sim_data.transfusion_cost:,.0f}.""")

    # ----- shared inputs for CEAC / CE-plane -----
    mean_daly = qaly_df["total_daly_averted_cum_mean"][-1:].values[0]
    min_daly = qaly_df["total_daly_averted_cum_lower"][-1:].values[0]
    max_daly = qaly_df["total_daly_averted_cum_upper"][-1:].values[0]
    sd_daly = (max_daly - min_daly) / (2 * 1.92)

    mean_costs_averted = stats_df["societal_savings_cum_no_death"][-1:].values[0]
    min_costs_averted = stats_df["societal_savings_cum_no_death_lower"][-1:].values[0]
    max_costs_averted = stats_df["societal_savings_cum_no_death_upper"][-1:].values[0]
    mean_screen_costs = stats_df["total_house_costs_cum"][-1:].values[0]

    n_sim = 10_000
    np.random.seed(12345)
    dalys_averted = np.random.normal(mean_daly, sd_daly, n_sim)

    x1 = mean_screen_costs - mean_costs_averted
    x2 = mean_screen_costs - min_costs_averted
    x3 = mean_screen_costs - max_costs_averted
    sd_x = abs((x3 - x2) / (2 * 1.96))
    costs_x = np.random.normal(x1, sd_x, n_sim)

    mean_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum"][-1:].values[0]
    min_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum_upper"][-1:].values[0]
    max_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum_lower"][-1:].values[0]
    sd_x_hs = abs((max_costs_averted_hs - min_costs_averted_hs) / (2 * 1.96))
    costs_x_hs = np.random.normal(mean_costs_averted_hs, sd_x_hs, n_sim)

    wtp = list(range(1, 10_001))

    with st.expander(f"Cost effectiveness acceptability curves - societal perspective: {region}"):
        st.latex(r"""NMB = (E * \lambda) - C""")
        ceac_soc, ratios_soc = compute_ceac(costs_x, dalys_averted, wtp)
        wtp_at_50 = np.interp(0.5, ceac_soc, wtp)
        prop_ce_base = float(np.mean(ratios_soc < updated_sim_data.threshold))
        prop_ce_gdp = float(np.mean(ratios_soc < updated_sim_data.gdp_ppp))
        prop_ce_gdp3 = float(np.mean(ratios_soc < updated_sim_data.gdp_ppp_3))

        nmb_mean = (mean_daly * updated_sim_data.threshold) - (mean_screen_costs - mean_costs_averted)
        nmb_lower = (min_daly * updated_sim_data.threshold) - (mean_screen_costs - min_costs_averted)
        nmb_upper = (max_daly * updated_sim_data.threshold) - (mean_screen_costs - max_costs_averted)

        if wtp_at_50 < 8_100:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(wtp, ceac_soc)
            plt.suptitle(f"Cost-effectiveness acceptability curve: {region}")
            plt.title("Societal Perspective")
            plt.ylabel("Probability cost-effective")
            plt.xlabel("Willingness-To-Pay")
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.grid(axis="both", alpha=.4, linewidth=.2)
            xlims, ylims = ax.get_xlim(), ax.get_ylim()
            plt.annotate(f"WTP at 50% prob.: ${wtp_at_50:,.0f}",
                         xy=(xlims[0] + .7 * (xlims[1] - xlims[0]),
                             ylims[0] + .8 * (ylims[1] - ylims[0])))
            st.pyplot(fig)
            download_fig(fig, f"CEAC_Societal_{region}.png", "Download PNG")
            st.write(f"""The net monetary benefit (societal perspective) in {region} is \\${nmb_mean:,.2f}[\\${nmb_lower:,.2f}, \\${nmb_upper:,.2f}].
                    A positive NMB implies the intervention is cost-effective and might be worth pursuing given competing priorities.
                    This assumes a conservative cost-effectiveness threshold of \\${updated_sim_data.threshold}.
                    There is a {100 * prop_ce_base:,.1f}% chance the intervention is cost-effective at a CE threshold of \\${updated_sim_data.threshold:,.0f},
                    {100 * prop_ce_gdp:,.1f}% at a CET of \\${updated_sim_data.gdp_ppp}, and {100 * prop_ce_gdp3:,.1f}% at a
                    CET of \\${updated_sim_data.gdp_ppp_3}. See also the cost-effectiveness analysis plane below.""")
        else:
            st.write(f"""Housing screening in {region} is not cost-effective from a societal perspective using
                     conventional cost-effectiveness thresholds including the controversial 3X WHO GDP threshold.
                     Cost-effectiveness acceptability curves for {region} are therefore not produced.
                     The net monetary benefit (societal perspective) in {region} is \\${nmb_mean:,.2f}[\\${nmb_lower:,.2f}, \\${nmb_upper:,.2f}].
                    A negative NMB implies the intervention is not cost-effective.
                    This assumes a cost-effectiveness threshold of \\${updated_sim_data.threshold}.""")

    with st.expander(f"Cost-effectiveness acceptability curves health sector perspective: {region}"):
        ceac_hs, ratios_hs = compute_ceac(costs_x_hs, dalys_averted, wtp)
        wtp_at_50 = np.interp(0.5, ceac_hs, wtp)
        if wtp_at_50 < 8_000:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(wtp, ceac_hs, label="CEAC")
            plt.suptitle(f"Cost-effectiveness acceptability curve: {region}")
            plt.title("Health system perspective")
            plt.ylabel("Probability cost-effective")
            plt.xlabel("Willingness-to-pay ($)")
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.grid(axis="both", alpha=.4, linewidth=.2)
            plt.legend(loc="center right")
            plt.hlines(y=0.5, xmin=0, xmax=wtp_at_50, color="red", linestyle="--", alpha=.7)
            plt.axvline(x=wtp_at_50, ymin=0.05, ymax=0.5 / plt.ylim()[1],
                        color='red', linestyle='--', alpha=0.7)
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_position(("data", 0))
            ax.plot(0, 1.01, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
            plt.annotate(f'WTP at 50% prob. = ${wtp_at_50:,.0f}', xy=(wtp_at_50, 0.5),
                         xytext=(15, 10), textcoords="offset points")
            st.pyplot(fig)
            download_fig(fig, f"CEAC_HS_{region}.png", "Download PNG")
        else:
            st.write(f"""Housing screening in {region} is not cost-effective from a healthcare perspective using
                     conventional cost-effectiveness thresholds including the controversial 3X WHO GDP threshold.
                     Cost-effectiveness acceptability curves for {region} are therefore not produced.""")

    with st.expander(f"Cost-effectiveness plane -- societal perspective: {region}"):
        fig = plot_ce_plane(dalys_averted, costs_x, region, "Societal Perspective", updated_sim_data)
        st.pyplot(fig)
        download_fig(fig, f"CEPlane_Societal_{region}.png", "Download PNG")
        prop_139 = np.mean(costs_x / dalys_averted < updated_sim_data.threshold)
        prop_2399 = np.mean(costs_x / dalys_averted < updated_sim_data.gdp_ppp)
        prop_7197 = np.mean(costs_x / dalys_averted < updated_sim_data.gdp_ppp_3)
        st.write(f"""Using a societal perspective and a subsidy rate of {params.subsidy * 100}%, there is {np.round(prop_139, 2)} probability the intervention is cost-effective in {region} using a threshold of \\${updated_sim_data.threshold}, {np.round(prop_2399,2)} probability using a threshold of \\${updated_sim_data.gdp_ppp}, and {np.round(prop_7197,2)} probability using a threshold of \\${updated_sim_data.gdp_ppp_3}. Note, while there are cost savings to households arising from subsidies, these costs are still incurred by governments/society and are included in the final calculations.""")

    with st.expander(f"Cost-effectiveness plane -- health system perspective: {region}"):
        fig = plot_ce_plane(dalys_averted, costs_x_hs, region, "Health System Perspective", updated_sim_data)
        st.pyplot(fig)
        download_fig(fig, f"CEPlane_HS_{region}.png", "Download PNG")
        prop_139 = np.mean(costs_x_hs / dalys_averted < updated_sim_data.threshold)
        prop_2399 = np.mean(costs_x_hs / dalys_averted < updated_sim_data.gdp_ppp)
        prop_7197 = np.mean(costs_x_hs / dalys_averted < updated_sim_data.gdp_ppp_3)
        st.write(f"""Using a health system perspective and a subsidy rate of {params.subsidy * 100}%, there is {np.round(prop_139,2)} probability the intervention is cost-effective in {region} using a threshold of \\${updated_sim_data.threshold}, {np.round(prop_2399,2)} probability using a threshold of \\${updated_sim_data.gdp_ppp}, and {np.round(prop_7197,2)} probability using a threshold of \\${updated_sim_data.gdp_ppp_3}.""")
        st.dataframe(pd.DataFrame({"costs": costs_x_hs, "dalys": dalys_averted}))

    with st.expander("Tornado Diagram - Sensitivity Analysis"):
        st.write("**Tornado diagrams show which parameters have the greatest impact on the net monetary benefit.**")
        st.write("Parameters are ranked by their impact magnitude. Red bars show unfavorable changes, green bars show favorable changes.")
        if st.button("Generate Tornado Analysis", key="tornado_btn"):
            with st.spinner("Running sensitivity analysis..."):
                tornado_results, base_nmb = tornado_analysis(params, updated_region_data,
                                                             region, updated_sim_data)
                tornado_df = pd.DataFrame(tornado_results)
                tornado_df['Impact Range'] = tornado_df['nmb_range'].apply(lambda x: f"${x:,.0f}")
                tornado_df['Base Value'] = tornado_df['base_value'].apply(lambda x: f"{x:.3f}")
                display_df = tornado_df[['parameter', 'Base Value', 'Impact Range']].copy()
                display_df.columns = ['Parameter', 'Base Value', 'Impact Range ($)']
                st.dataframe(display_df.style)

                fig = plot_tornado_diagram(tornado_results, base_nmb, region)
                st.pyplot(fig)
                download_fig(fig, f"Tornado_Diagram_{region}.png", "Download Tornado Diagram")

                most_sensitive = tornado_results[0]
                st.write(f"**Most sensitive parameter:** {most_sensitive['parameter']} with an impact range of ${most_sensitive['nmb_range']:,.0f}")
                st.write(f"**Base case Net Monetary Benefit:** ${base_nmb:,.0f}")

# =====================================================================
# Tab 3: National summary and notes
# =====================================================================
with tabs[3]:
    with st.expander("View Combined National Level Data"):
        st.dataframe(final_df)
    with st.expander("Additional Notes"):
        additional_notes = """
        1. 32.8% of severe malaria anemia patients get blood transfusion. [Ackerman et. al (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611367/)
        2. $72: adjusted costs of blood transfusion in Uganda. [Watson et al (1990)](https://www.sciencedirect.com/science/article/abs/pii/0955388690900892). This is cost of blood collection and processing.
        3. Length of stay; Median 4 days [Machini et al (2022)](https://bmjopen.bmj.com/content/12/6/e059263.long)
        """
        st.write(additional_notes)
#======================================================================
# Tab 4: Subsidy Analysis
#======================================================================

with tabs[4]:
    st.subheader("Subsidy policy analysis: adoption, financing, and equity")
    st.write(
        "This section treats subsidies as financing transfers. The real "
        "resource cost of modifying a house is unchanged; the subsidy changes "
        "the household-facing price, adoption, and the distribution of costs "
        "between households, government, and suppliers."
    )

    def _fmt_money(value):
        if value is None or pd.isna(value):
            return "NA"
        return f"${value:,.0f}"

    def _fmt_number(value, decimals=0):
        if value is None or pd.isna(value):
            return "NA"
        return f"{value:,.{decimals}f}"

    def _fmt_pct(value):
        if value is None or pd.isna(value):
            return "NA"
        return f"{value:,.1f}%"

    def _size_for_bubble(values, min_size=80, max_size=900):
        arr = np.asarray(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0 or np.max(arr) <= 0:
            return np.full_like(arr, min_size, dtype=float)
        return min_size + (max_size - min_size) * arr / np.max(arr)

    def _dominated_policies(df, x_col, y_col):
        dominated = []
        for a in df.index:
            axv = float(df.loc[a, x_col])
            ayv = float(df.loc[a, y_col])
            for b in df.index:
                if a == b:
                    continue
                bxv = float(df.loc[b, x_col])
                byv = float(df.loc[b, y_col])
                weakly_better = bxv <= axv and byv >= ayv
                strictly_better = bxv < axv or byv > ayv
                if weakly_better and strictly_better:
                    dominated.append(a)
                    break
        return dominated

    with st.expander("How to interpret this section", expanded=False):
        st.markdown(
            """
            **Budget impact** uses undiscounted cash outlays because annual budgets are cash constraints.  
            **Cost-effectiveness** uses discounted costs, savings, and DALYs.  
            **Deadweight subsidy** is public transfer paid to households that would likely have adopted even without subsidy.  
            **Supplier capture/leakage** appears when pass-through is below 100%; it is public money that does not reduce the household price.  
            **Targeting is fractional**, because the dataset contains regional poverty shares rather than household-level poverty status.
            """
        )

    # --- scenario controls ---
    col_a, col_b = st.columns(2)

    with col_a:
        n_pol_sims = st.slider(
            "Simulations per policy", 100, 3000, 500, step=100,
            help="Common random numbers are used across policies."
        )
        poverty_var = st.selectbox(
            "Poverty variable for targeting", ["mdp", "subsistence", "pdm"]
        )
        transmission_threshold = st.slider(
            "High-transmission prevalence threshold", 0.01, 0.50, 0.10,
            step=0.01, format="%.2f"
        )
        admin_cost = st.number_input(
            "Admin/targeting cost per subsidized household ($)",
            0.0, 100.0, 0.0, step=1.0
        )
        pass_through = st.slider(
            "Subsidy pass-through", 0.0, 1.0, 1.0, step=0.05,
            help="Share of the public subsidy that lowers the household-facing price."
        )
        annual_budget_cap = st.number_input(
            "Annual national subsidy budget cap ($, 0 = uncapped)",
            0.0, 1_000_000_000.0, 0.0, step=1_000_000.0
        )
        allocation_rule = st.selectbox(
            "Allocation rule if budget cap binds",
            ["proportional", "poor_first", "high_transmission_first", "cea_first"]
        )

    with col_b:
        reference_price = national_reference_price(params, regions)
        p_full_price = st.slider(
            f"Adoption at full reference price (${reference_price:,.0f})",
            0.01, 0.80, 0.20, step=0.01,
            help="This replaces the previous hard-coded adoption coefficients."
        )
        p_free = st.slider(
            "Adoption if household price is $0", 0.05, 0.99, 0.85,
            step=0.01
        )
        poor_price_multiplier = st.slider(
            "Poor-household price multiplier", 0.0, 2.0, 0.0,
            step=0.05,
            help="0 means no extra affordability penalty; 0.5 means poor households behave as if price were 50% higher."
        )
        st.caption(
            "Adoption is calibrated by two interpretable points: uptake at the "
            "full reference price and uptake if the intervention is free."
        )

    adoption = AdoptionCalibration(
        p_full_price=p_full_price,
        p_free=p_free,
        reference_price_usd=reference_price,
        poor_price_multiplier=poor_price_multiplier,
    )

    budget_cap = annual_budget_cap if annual_budget_cap > 0 else None
    scenarios = default_scenarios(
        poverty_var=poverty_var,
        transmission_threshold=transmission_threshold,
        pass_through=pass_through,
        admin_cost_per_hh_usd=admin_cost,
        annual_budget_cap_usd=budget_cap,
        allocation_rule=allocation_rule,
    )

    comparator_name = st.selectbox(
        "Comparator for incremental results", [s.name for s in scenarios], index=0
    )
    perspective = st.radio(
        "Incremental CEA perspective",
        ["societal", "government", "program", "household"],
        horizontal=True,
        help=(
            "Government subtracts health-system savings from public outlay; "
            "program counts public outlay only; societal excludes transfers "
            "but includes real resource and admin costs."
        )
    )
    focus_policy_name = st.selectbox(
        "Policy for annual fiscal and deadweight visuals", [s.name for s in scenarios], index=0
    )

    # ---------- Targeting rationale before simulation ----------
    st.markdown("#### Where should subsidy be targeted?")
    targeting_df = pd.DataFrame(
        regional_targeting_rows(regions, poverty_var, transmission_threshold)
    )
    if not targeting_df.empty:
        top_priority = targeting_df.sort_values(
            "Targeting priority index", ascending=False
        ).head(3)["Region"].tolist()
        st.write(
            "The bubble plot frames the targeting problem before the Monte Carlo "
            "simulation: the strongest candidates combine high malaria prevalence, "
            "high poverty, and many modifiable households. Under these inputs, the "
            f"highest priority regions by the simple burden-poverty index are **{', '.join(top_priority)}**."
        )

        fig_target, ax_target = plt.subplots(figsize=(10, 6))
        sizes = _size_for_bubble(targeting_df["Modifiable households"], 80, 900)
        ax_target.scatter(
            targeting_df["Malaria prevalence (%)"],
            targeting_df["Poverty share (%)"],
            s=sizes,
            alpha=0.55,
        )
        for _, row in targeting_df.iterrows():
            ax_target.annotate(
                row["Region"],
                (row["Malaria prevalence (%)"], row["Poverty share (%)"]),
                xytext=(4, 3), textcoords="offset points", fontsize=7
            )
        ax_target.axvline(transmission_threshold * 100.0, linestyle="--", linewidth=1)
        ax_target.axhline(
            targeting_df["Median poverty share (%)"].iloc[0],
            linestyle="--", linewidth=1
        )
        ax_target.set_xlabel("Malaria prevalence (%)")
        ax_target.set_ylabel(f"Poverty share: {poverty_var.upper()} (%)")
        ax_target.set_title("Regional subsidy-targeting priority map")
        ax_target.grid(alpha=.3, linewidth=.3)
        for sp in ["top", "right"]:
            ax_target.spines[sp].set_visible(False)
        st.pyplot(fig_target)
        download_fig(fig_target, "Subsidy_Targeting_Priority_Map.png", "Download targeting chart")

        with st.expander("Regional targeting inputs", expanded=False):
            st.dataframe(
                targeting_df.set_index("Region").style.format({
                    "Malaria prevalence (%)": "{:,.1f}%",
                    "Poverty share (%)": "{:,.1f}%",
                    "Modifiable households": "{:,.0f}",
                    "2024 RDT-positive cases": "{:,.0f}",
                    "Targeting priority index": "{:,.0f}",
                    "Median poverty share (%)": "{:,.1f}%",
                    "Median malaria prevalence (%)": "{:,.1f}%",
                })
            )

    simulation_settings = {
        "n_pol_sims": int(n_pol_sims),
        "poverty_var": poverty_var,
        "transmission_threshold": float(transmission_threshold),
        "admin_cost": float(admin_cost),
        "pass_through": float(pass_through),
        "annual_budget_cap": float(annual_budget_cap),
        "allocation_rule": allocation_rule,
        "p_full_price": float(p_full_price),
        "p_free": float(p_free),
        "poor_price_multiplier": float(poor_price_multiplier),
        "reference_price": float(reference_price),
    }

    if st.button("Run subsidy policy analysis", key="policy_btn"):
        with st.spinner("Running policy scenarios with common random numbers..."):
            draws_by_policy = {
                s.name: run_policy_simulations(
                    params, regions, s, updated_sim_data,
                    n_simulations=n_pol_sims, adoption=adoption
                )
                for s in scenarios
            }
            st.session_state["subsidy_policy_results"] = {
                "draws_by_policy": draws_by_policy,
                "scenarios": scenarios,
                "settings": simulation_settings,
            }

    if "subsidy_policy_results" in st.session_state:
        stored = st.session_state["subsidy_policy_results"]
        draws_by_policy = stored["draws_by_policy"]
        stored_scenarios = stored["scenarios"]
        stored_settings = stored.get("settings", {})
        display_budget_cap_value = float(stored_settings.get("annual_budget_cap", annual_budget_cap))
        display_budget_cap = display_budget_cap_value if display_budget_cap_value > 0 else None
        if stored_settings != simulation_settings:
            st.warning(
                "Displayed subsidy results use the previous scenario settings. "
                "Run the subsidy policy analysis again to refresh the outputs."
            )

        available_names = list(draws_by_policy.keys())
        if comparator_name not in available_names:
            comparator_name = available_names[0]
        if focus_policy_name not in available_names:
            focus_policy_name = available_names[0]

        bia_rows = [budget_impact_row(draws_by_policy[s.name], s, updated_sim_data)
                    for s in stored_scenarios if s.name in draws_by_policy]
        bia_df = pd.DataFrame(bia_rows).set_index("Policy")

        nmb_df = policy_nmb_table(
            draws_by_policy, updated_sim_data, perspective=perspective,
            threshold=updated_sim_data.threshold
        )
        best_policy = nmb_df.index[0]
        best_bia = bia_df.loc[best_policy]
        best_nmb = nmb_df.loc[best_policy]

        # ---------- Executive scorecard ----------
        st.markdown("#### Executive policy scorecard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Highest expected NMB", best_policy)
        c2.metric("Expected NMB", _fmt_money(best_nmb["Expected NMB ($)"]))
        c3.metric("Probability best", _fmt_pct(best_nmb["Probability best (%)"]))
        c4, c5, c6 = st.columns(3)
        c4.metric("Government outlay, cash", _fmt_money(best_bia["Government outlay, cash ($)"]))
        c5.metric("Net fiscal impact, NPV", _fmt_money(best_bia["Net fiscal impact, NPV ($)"]))
        c6.metric("Poor households reached", _fmt_number(best_bia["Poor households reached"]))

        offset = (best_bia["Health-system savings, NPV ($)"] /
                  best_bia["Government outlay, NPV ($)"]
                  if best_bia["Government outlay, NPV ($)"] > 0 else np.nan)
        st.success(
            f"At ${updated_sim_data.threshold:,.0f} per DALY, **{best_policy}** has the "
            f"highest expected net monetary benefit from the **{perspective}** perspective. "
            f"It averts an estimated **{_fmt_number(best_bia['DALYs averted'], 1)} DALYs**, "
            f"reaches **{_fmt_number(best_bia['Poor households reached'])} poor households**, "
            f"and has an estimated NPV net fiscal impact of **{_fmt_money(best_bia['Net fiscal impact, NPV ($)'])}**. "
            f"Health-system savings offset **{_fmt_pct(100.0 * offset) if not pd.isna(offset) else 'NA'}** "
            "of discounted public outlay."
        )

        # ---------- Main tables ----------
        with st.expander("Budget impact, targeting, and subsidy efficiency table", expanded=True):
            money_cols = [c for c in bia_df.columns if "($)" in c or "cost" in c.lower()
                          or "impact" in c.lower() or "subsidy" in c.lower()
                          or "transfer" in c.lower() or "capture" in c.lower()]
            percent_cols = [c for c in bia_df.columns if "(%)" in c]
            count_cols = [c for c in bia_df.columns if c not in money_cols + percent_cols]
            fmt = {c: "${:,.0f}" for c in money_cols}
            fmt.update({c: "{:,.1f}%" for c in percent_cols})
            fmt.update({c: "{:,.0f}" for c in count_cols})
            st.dataframe(bia_df.style.format(fmt))
            st.caption(
                "Cash outlay is undiscounted for budget planning. NPV columns are discounted "
                "and should be used for economic evaluation."
            )

        with st.expander("Expected net benefit ranking", expanded=True):
            nmb_fmt = {
                "Mean DALYs": "{:,.1f}",
                "Mean cost, NPV ($)": "${:,.0f}",
                "Expected NMB ($)": "${:,.0f}",
                "NMB lower 95% ($)": "${:,.0f}",
                "NMB upper 95% ($)": "${:,.0f}",
                "Probability best (%)": "{:,.1f}%",
            }
            st.dataframe(nmb_df.style.format(nmb_fmt))

        # ---------- Annual budget impact ----------
        st.markdown(f"#### Annual affordability profile: {focus_policy_name}")
        annual_df = pd.DataFrame(
            annual_budget_impact_rows(draws_by_policy[focus_policy_name])
        ).set_index("Year")
        peak_year = int(annual_df["Government outlay, cash ($)"].idxmax())
        peak_outlay = float(annual_df["Government outlay, cash ($)"].max())
        total_cash = float(annual_df["Government outlay, cash ($)"].sum())
        total_hs_cash = float(annual_df["Health-system savings, cash ($)"].sum())
        cash_offset = total_hs_cash / total_cash if total_cash > 0 else np.nan
        st.write(
            f"For **{focus_policy_name}**, peak government cash outlay occurs in model year "
            f"**{peak_year}** at **{_fmt_money(peak_outlay)}**. Across the horizon, "
            f"health-system savings offset **{_fmt_pct(100.0 * cash_offset) if not pd.isna(cash_offset) else 'NA'}** "
            "of undiscounted public outlay."
        )

        fig_ann, ax_ann = plt.subplots(figsize=(10, 6))
        ax_ann.plot(annual_df.index, annual_df["Government outlay, cash ($)"],
                    label="Government outlay")
        ax_ann.plot(annual_df.index, annual_df["Health-system savings, cash ($)"],
                    label="Health-system savings")
        ax_ann.plot(annual_df.index, annual_df["Net fiscal impact, cash ($)"],
                    label="Net fiscal impact")
        if display_budget_cap is not None:
            ax_ann.axhline(display_budget_cap, linestyle="--", linewidth=1,
                           label="Annual budget cap")
        ax_ann.set_xlabel("Model year")
        ax_ann.set_ylabel("Cash flow ($)")
        ax_ann.set_title(f"Annual fiscal flows: {focus_policy_name}")
        ax_ann.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        ax_ann.grid(alpha=.3, linewidth=.3)
        for sp in ["top", "right"]:
            ax_ann.spines[sp].set_visible(False)
        ax_ann.legend(loc="best")
        st.pyplot(fig_ann)
        download_fig(fig_ann, "Subsidy_Annual_Fiscal_Flows.png", "Download annual fiscal chart")

        with st.expander("Annual budget-impact table", expanded=False):
            annual_money_cols = [c for c in annual_df.columns if "($)" in c]
            annual_fmt = {c: "${:,.0f}" for c in annual_money_cols}
            annual_fmt.update({
                "Subsidized households": "{:,.0f}",
                "Additional adopters": "{:,.0f}",
                "Cases averted": "{:,.0f}",
                "Probability cap binds": "{:.1%}",
            })
            st.dataframe(annual_df.style.format(annual_fmt))

        # ---------- Efficiency frontier ----------
        st.markdown("#### Efficiency frontier: health gain versus fiscal cost")
        x_col = "Net fiscal impact, NPV ($)"
        y_col = "DALYs averted"
        frontier_df = bia_df[[x_col, y_col, "Poor households reached"]].copy()
        fig_frontier, ax_frontier = plt.subplots(figsize=(10, 6))
        sizes = _size_for_bubble(frontier_df["Poor households reached"], 90, 950)
        ax_frontier.scatter(frontier_df[x_col], frontier_df[y_col], s=sizes, alpha=0.55)
        for name, row in frontier_df.iterrows():
            ax_frontier.annotate(
                name, (row[x_col], row[y_col]), xytext=(4, 3),
                textcoords="offset points", fontsize=7
            )
        ax_frontier.axvline(0, linewidth=1)
        ax_frontier.set_xlabel("Net fiscal impact, NPV ($)")
        ax_frontier.set_ylabel("DALYs averted")
        ax_frontier.set_title("Policy efficiency frontier")
        ax_frontier.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        ax_frontier.grid(alpha=.3, linewidth=.3)
        for sp in ["top", "right"]:
            ax_frontier.spines[sp].set_visible(False)
        st.pyplot(fig_frontier)
        download_fig(fig_frontier, "Subsidy_Efficiency_Frontier.png", "Download frontier chart")

        dominated = _dominated_policies(frontier_df, x_col, y_col)
        if dominated:
            st.caption(
                "Weakly dominated on this frontier: " + ", ".join(dominated) +
                ". A dominated policy costs at least as much and averts no more DALYs than another option."
            )
        else:
            st.caption(
                "No policy is weakly dominated using net fiscal impact and DALYs averted alone."
            )

        # ---------- Equity chart ----------
        st.markdown("#### Equity reach")
        equity_df = bia_df.sort_values("Poor households reached")
        fig_equity, ax_equity = plt.subplots(figsize=(10, 7))
        ax_equity.barh(equity_df.index, equity_df["Poor households reached"])
        ax_equity.set_xlabel("Poor households reached")
        ax_equity.set_title("Equity reach by subsidy policy")
        ax_equity.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax_equity.grid(axis="x", alpha=.3, linewidth=.3)
        for sp in ["top", "right"]:
            ax_equity.spines[sp].set_visible(False)
        st.pyplot(fig_equity)
        download_fig(fig_equity, "Subsidy_Equity_Reach.png", "Download equity chart")

        st.write(
            f"The recommended policy reaches **{_fmt_pct(best_bia['Poor share of adopters (%)'])}** "
            "poor households among adopters. If the policy objective is distributional equity rather "
            "than expected NMB, compare this chart against the efficiency frontier before choosing."
        )

        # ---------- Deadweight and leakage decomposition ----------
        st.markdown(f"#### Deadweight and leakage decomposition: {focus_policy_name}")
        decomp_df = pd.DataFrame(
            fiscal_decomposition_from_budget_row(bia_df.loc[focus_policy_name])
        )
        total_decomp = float(decomp_df["Value ($)"].sum())
        if total_decomp > 0:
            fig_decomp, ax_decomp = plt.subplots(figsize=(10, 5))
            ax_decomp.barh(decomp_df["Component"], decomp_df["Value ($)"])
            ax_decomp.set_xlabel("Government outlay, NPV ($)")
            ax_decomp.set_title("Where public subsidy dollars go")
            ax_decomp.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
            ax_decomp.grid(axis="x", alpha=.3, linewidth=.3)
            for sp in ["top", "right"]:
                ax_decomp.spines[sp].set_visible(False)
            st.pyplot(fig_decomp)
            download_fig(fig_decomp, "Subsidy_Deadweight_Leakage_Decomposition.png", "Download decomposition chart")
            st.dataframe(decomp_df.set_index("Component").style.format({"Value ($)": "${:,.0f}"}))
        else:
            st.info("The selected focus policy has no public outlay to decompose.")

        dead_share = bia_df.loc[focus_policy_name, "Deadweight share of transfers (%)"]
        leakage = bia_df.loc[focus_policy_name, "Supplier capture/leakage, NPV ($)"]
        st.write(
            f"For **{focus_policy_name}**, the model estimates a deadweight share of "
            f"**{_fmt_pct(dead_share)}** of transfers and supplier capture/leakage of "
            f"**{_fmt_money(leakage)}**. High values here mean the subsidy is less well targeted "
            "or not fully passed through to households."
        )

        # ---------- Incremental uncertainty ----------
        st.markdown("#### Uncertainty: incremental net benefit and CEAC")
        comparator = draws_by_policy[comparator_name]
        inc_rows = []
        for s in stored_scenarios:
            if s.name == comparator_name or s.name not in draws_by_policy:
                continue
            inc_rows.append(
                incremental_summary_row(
                    s.name, draws_by_policy[s.name], comparator,
                    updated_sim_data.threshold, perspective=perspective
                )
            )
        inc_df = pd.DataFrame(inc_rows).set_index("Policy")
        if not inc_df.empty:
            st.dataframe(
                inc_df.style.format({
                    "Delta DALYs": "{:,.1f}",
                    "Delta cost, NPV ($)": "${:,.0f}",
                    "Mean INMB ($)": "${:,.0f}",
                    "P(INMB > 0)": "{:.1%}",
                    "Incremental cost per DALY ($)": "${:,.0f}",
                })
            )

        wtp = list(range(0, 10_001, 50))
        fig_ceac, ax_ceac = plt.subplots(figsize=(10, 6))
        for s in stored_scenarios:
            if s.name == comparator_name or s.name not in draws_by_policy:
                continue
            ceac, dE, dC = incremental_nmb_ceac(
                draws_by_policy[s.name], comparator,
                updated_sim_data, wtp, perspective=perspective
            )
            ax_ceac.plot(wtp, ceac, label=s.name)
        ax_ceac.axvline(updated_sim_data.threshold, linestyle="--", linewidth=1,
                        label=f"Threshold: ${updated_sim_data.threshold:,.0f}/DALY")
        ax_ceac.set_xlabel("Willingness-to-pay ($/DALY)")
        ax_ceac.set_ylabel("Probability cost-effective")
        ax_ceac.set_title(f"Incremental NMB-CEAC vs '{comparator_name}' ({perspective})")
        ax_ceac.grid(alpha=.3, linewidth=.3)
        for sp in ["top", "right"]:
            ax_ceac.spines[sp].set_visible(False)
        ax_ceac.legend(loc="best", fontsize=8)
        ax_ceac.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        st.pyplot(fig_ceac)
        download_fig(fig_ceac, "Subsidy_CEAC.png", "Download CEAC")

        with st.expander("Policy caveats for manuscript or briefing", expanded=False):
            st.markdown(
                f"""
                - The targeting model is fractional: the app splits each region using the selected poverty share (`{poverty_var}`), rather than identifying individual poor households.
                - Adoption response is calibrated to two assumptions: adoption at the full reference price and adoption when household price is zero. These should be replaced with pilot uptake, stated-preference, or revealed-preference evidence when available.
                - A subsidy is not treated as a reduction in societal resource cost. It changes adoption and payer distribution, while the full materials and labor cost remains in the economic evaluation.
                - Annual budget caps are applied as national annual caps. If the cap binds, the allocation rule determines who receives the subsidized voucher first.
                """
            )


# =====================================================================
# Tab 5: Contact
# =====================================================================
with tabs[5]:
    with st.expander("Contact us"):
        contact_form = """
            <form action="https://formsubmit.co/wochieng@cdc.gov" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <input type="text" name="_honey" style="display:none">
                <input type="hidden" name="_cc" value="ocu9@cdc.gov">
                <textarea name="message" placeholder="Details of your problem"></textarea>
                <button type="submit">Send Information</button>
            </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)

        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        local_css("style/style.css")