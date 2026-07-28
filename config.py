
# =====================================================================
# FILE: config.py
# Model configuration: simulation inputs (dataclass) and calibration.
# =====================================================================
import streamlit as st
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationInputs:
    um_mean: float = .3
    um_sd: float = .1
    severe_mean: float = .082
    severe_sd: float = .02
    severe_anemia_mean: float = .322 * severe_mean
    severe_anemia_sd: float = severe_anemia_mean * .1
    cerebral_malaria_mean: float = .002 * severe_mean
    cerebral_malaria_sd: float = cerebral_malaria_mean * .1
    cerebral_anemia: float = .001 * severe_mean
    cerebral_anemia_sd: float = cerebral_anemia * .1
    neurological: float = .098 * severe_mean
    neurological_sd: float = neurological * .1
    deaths: float = .001268
    deaths_sd: float = .000407
    los: float = 2.3
    los_sd: float = .1
    dw_um: float = .2078
    dw_um_sd: float = dw_um * .19
    dw_cerebral: float = .471
    dw_cerebral_sd: float = .047
    dw_sev_anemia: float = .149
    dw_sev_anemia_sd: float = .0149
    dw_cerebral_anemia: float = .620
    dw_cerebral_anemia_sd: float = .062
    duration_um: float = 5.1
    duration_um_sd: float = duration_um * .19
    duration_sm: float = 8.75
    duration_sm_sd: float = .875
    duration_sm_comp: float = 11.0
    duration_neurological: float = 10.1
    duration_neurological_sd: float = .101
    duration_severe_anemia: float = 11.0
    duration_severe_anemia_sd: float = 1.1
    duration_cerebral: float = 6.5
    duration_cerebral_sd: float = .65
    duration_sm_overall: float = duration_sm + duration_sm_comp
    duration_sm_overall_sd: float = duration_sm_overall * .19
    dw_sm: float = .471
    dw_sm_sd: float = (.550 - .411) / (3.92 * .471)
    opd_cost: float = 5.84
    opd_cost_sd: float = .4975
    ipd_cost: float = 19.77
    ipd_cost_sd: float = 2.615
    transfusion_cost: float = 86.25
    prop_sam_transfused: float = .328
    hematinics: float = 16.16
    hh_um_cost: float = 12.65
    hh_um_cost_sd: float = 1.145
    hh_sm_cost: float = 20.29
    hh_sm_cost_sd: float = 2.9975
    mean_vsly: float = 5764.17
    std_vsly: float = 2041.24
    disc_lifespan: float = 28.595
    threshold: float = 139.0
    gdp_ppp: float = 2399
    gdp_ppp_3: float = gdp_ppp * 3


class Parameters:
    # annual_repair_rate is passed in (was a module global before). app.py
    # computes it from the selected region and hands it over at construction.
    def __init__(self, annual_repair_rate=0.075):
        self.subsidy = .0001
        self.years = np.arange(0, 10)
        self.max_coverage = 0.80
        self.growth_rate = 1.5
        self.midpoint_year = 5.0
        self.shape = 0.4
        self.baseline_prevalence = 0.30
        self.efficacy = 0.32
        self.efficacy_std = 0.0625
        self.daly_per_case = 1.2
        self.failure_rate = 0.1
        self.annual_repair_rate = annual_repair_rate
        self.repair_cost_fraction = 0.3
        self.discount_rate = 0.03
        self.cost_mud = {"mean": 101, "std": 2.75}
        self.cost_brick = {"mean": 150, "std": 4.25}

    def update_from_sliders(self):
        with st.sidebar.expander("🔧 Calibration Sliders"):
            self.subsidy = st.slider("Subsidy rate", .0001, .9999, self.subsidy,
                help="Select subsidy rate for analyses. Bounds .0001-.9999 avoid zero-division.")
            self.max_coverage = st.slider("Max Coverage", 0.0, 1.0, self.max_coverage,
                help="Maximum proportion of houses that can be modified")
            self.growth_rate = st.slider("Growth Rate", 1.1, 5.0, self.growth_rate,
                help="Controls speed of sigmoid scale-up")
            self.shape = st.slider("Growth Curve Shape", 0.2, 2.0, float(self.shape),
                help="Control shape of scale-up growth curve")
            self.midpoint_year = st.slider("Midpoint Year", 0, 10, int(self.midpoint_year),
                help="Year at which 50% of scale-up is achieved")
            self.baseline_prevalence = st.slider("Baseline Malaria Prevalence", 0.0, 1.0,
                self.baseline_prevalence, help="Proportion with malaria before intervention")
            self.efficacy = st.slider("Efficacy", 0.0, 1.0, self.efficacy,
                help="Effectiveness of house modification at reducing malaria")
            self.efficacy_std = st.slider("Efficacy Std Dev", 0.0, 0.2, self.efficacy_std,
                help="Standard deviation around efficacy estimate")
            self.daly_per_case = st.slider("DALYs per Case Averted", 0.1, 10.0, self.daly_per_case,
                help="Disability-Adjusted Life Years lost per case")
            self.failure_rate = st.slider("Annual Screen Failure Rate", 0.0, 0.5, self.failure_rate,
                help="Annual proportion of screens that get damaged")
            self.annual_repair_rate = st.slider("Annual Repair Rate", 0.0, 1.0, self.annual_repair_rate,
                help="Proportion of failed screens repaired annually")
            self.repair_cost_fraction = st.slider("Repair Cost Fraction", 0.0, 1.0,
                self.repair_cost_fraction, help="Repair cost as a fraction of initial cost")
            self.discount_rate = st.slider("Discount Rate", 0.0, 0.1, self.discount_rate,
                help="Discount rate used for economic evaluation")
            self.cost_mud["mean"] = st.slider("Mud Wall Cost Mean", 50, 200, self.cost_mud["mean"],
                help="Mean cost for modifying a mud-walled house")
            self.cost_mud["std"] = st.slider("Mud Wall Cost Std Dev", 0.0, 20.0, self.cost_mud["std"],
                help="Standard deviation for mud wall cost")
            self.cost_brick["mean"] = st.slider("Brick Wall Cost Mean", 50, 300, self.cost_brick["mean"],
                help="Mean cost for modifying a brick-walled house")
            self.cost_brick["std"] = st.slider("Brick Wall Cost Std Dev", 0.0, 20.0,
                self.cost_brick["std"], help="Standard deviation for brick wall cost")

