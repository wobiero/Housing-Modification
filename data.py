# =====================================================================
# FILE: data.py
# Static census data, labels, and derived economic constants.
# =====================================================================
"""No dependencies on other project modules -- pure data."""

discount_rate = 0.03
min_wage_ppp = 1298  # Source: Wikipedia
human_cap = (min_wage_ppp
             * ((1 - (1 + discount_rate) ** (-53)) / discount_rate)
             * (1 + discount_rate) ** (-12.5))

terms_conditions = """

"""

regions = {
    "Kampala": {"population": 1_797_722, "households": 529_057, "bricks": .635, "mud": .017,
                "other_wall": .348, "subsistence": .017, "pdm": .692, "mal_prevalence": .01,
                "pop_growth_rate": .023, "pop_u5": 173_460, "mdp": .10, "household_size": 2.9,
                "modifiable_homes": .8175, "rdt_pos_24": 62_854},
    "Buganda": {"population": 11_171_924, "households": 2_894_776, "bricks": .66, "mud": .09,
                "other_wall": .25, "subsistence": .173, "pdm": .227, "mal_prevalence": .05,
                "pop_growth_rate": .034, "pop_u5": 1_510_794, "mdp": .31, "household_size": 3.6,
                "modifiable_homes": .83, "rdt_pos_24": 826_809},
    "Busoga": {"population": 4_363_295, "households": 965_299, "bricks": .716, "mud": .11,
               "other_wall": .174, "subsistence": .38, "pdm": .143, "mal_prevalence": .21,
               "pop_growth_rate": .021, "pop_u5": 659_016, "mdp": .61, "household_size": 4.4,
               "modifiable_homes": .858, "rdt_pos_24": 1_116_021},
    "Bukedi": {"population": 2_372_489, "households": 503_727, "bricks": .692, "mud": .195,
               "other_wall": .113, "subsistence": .499, "pdm": .184, "mal_prevalence": .03,
               "pop_growth_rate": .024, "pop_u5": 377_096, "mdp": .78, "household_size": 4.7,
               "modifiable_homes": .846, "rdt_pos_24": 650_086},
    "Bugisu": {"population": 1_827_757, "households": 446_015, "bricks": .308, "mud": .563,
               "other_wall": .128, "subsistence": .412, "pdm": .442, "mal_prevalence": .05,
               "pop_growth_rate": .024, "pop_u5": 252_535, "mdp": .72, "household_size": 4.0,
               "modifiable_homes": .6545, "rdt_pos_24": 236_052},
    "Sebei": {"population": 377_294, "households": 80_679, "bricks": .05, "mud": .845,
              "other_wall": .105, "subsistence": .406, "pdm": .763, "mal_prevalence": .18,
              "pop_growth_rate": .024, "pop_u5": 59_147, "mdp": .76, "household_size": 4.7,
              "modifiable_homes": .525, "rdt_pos_24": 22_944},
    "Teso": {"population": 2_462_387, "households": 489_620, "bricks": .877, "mud": .03,
             "other_wall": .092, "subsistence": .473, "pdm": .296, "mal_prevalence": .08,
             "pop_growth_rate": .031, "pop_u5": 393_460, "mdp": .50, "household_size": 4.9,
             "modifiable_homes": .878, "rdt_pos_24": 796_777},
    "Karamoja": {"population": 1_496_117, "households": 313_987, "bricks": .204, "mud": .613,
                 "other_wall": .183, "subsistence": .709, "pdm": .195, "mal_prevalence": .34,
                 "pop_growth_rate": .042, "pop_u5": 279_291, "mdp": .76, "household_size": 4.7,
                 "modifiable_homes": .602, "rdt_pos_24": 376_778},
    "Lango": {"population": 2_546_116, "households": 575_559, "bricks": .822, "mud": .055,
              "other_wall": .123, "subsistence": .476, "pdm": .178, "mal_prevalence": .13,
              "pop_growth_rate": .023, "pop_u5": 372_642, "mdp": .36, "household_size": 4.4,
              "modifiable_homes": .911, "rdt_pos_24": 732_083},
    "Acholi": {"population": 2_044_355, "households": 466_128, "bricks": .836, "mud": .035,
               "other_wall": .129, "subsistence": .50, "pdm": .218, "mal_prevalence": .12,
               "pop_growth_rate": .032, "pop_u5": 306_357, "mdp": .69, "household_size": 4.3,
               "modifiable_homes": .918, "rdt_pos_24": 1_056_613},
    "West Nile": {"population": 3_316_255, "households": 646_361, "bricks": .779, "mud": .103,
                  "other_wall": .119, "subsistence": .535, "pdm": .182, "mal_prevalence": .22,
                  "pop_growth_rate": .039, "pop_u5": 510_018, "mdp": .76, "household_size": 5.1,
                  "modifiable_homes": .889, "rdt_pos_24": 535_071},
    "Madi": {"population": 553_145, "households": 108_262, "bricks": .838, "mud": .042,
             "other_wall": .121, "subsistence": .473, "pdm": .233, "mal_prevalence": .22,
             "pop_growth_rate": .039, "pop_u5": 78_127, "mdp": .76, "household_size": 5.0,
             "modifiable_homes": .9185, "rdt_pos_24": 126_988},
    "Bunyoro": {"population": 2_792_732, "households": 663_258, "bricks": .487, "mud": .359,
                "other_wall": .155, "subsistence": .314, "pdm": .206, "mal_prevalence": .09,
                "pop_growth_rate": .033, "pop_u5": 454_135, "mdp": .42, "household_size": 4.2,
                "modifiable_homes": .743, "rdt_pos_24": 310_050},
    "Tooro": {"population": 2_154_161, "households": 504_035, "bricks": .303, "mud": .555,
              "other_wall": .142, "subsistence": .33, "pdm": .237, "mal_prevalence": .05,
              "pop_growth_rate": .028, "pop_u5": 330_935, "mdp": .40, "household_size": 4.2,
              "modifiable_homes": .6515, "rdt_pos_24": 246_116},
    "Rwenzori": {"population": 1_233_467, "households": 272_449, "bricks": .514, "mud": .353,
                 "other_wall": .133, "subsistence": .367, "pdm": .299, "mal_prevalence": .01,
                 "pop_growth_rate": .026, "pop_u5": 192_861, "mdp": .35, "household_size": 4.4,
                 "modifiable_homes": .757, "rdt_pos_24": 102_010},
    "Ankole": {"population": 3_608_968, "households": 842_783, "bricks": .458, "mud": .376,
               "other_wall": .166, "subsistence": .278, "pdm": .308, "mal_prevalence": .03,
               "pop_growth_rate": .022, "pop_u5": 448_734, "mdp": .30, "household_size": 4.2,
               "modifiable_homes": .729, "rdt_pos_24": 290_263},
    "Kigezi": {"population": 1_787_231, "households": 396_918, "bricks": .292, "mud": .541,
               "other_wall": .168, "subsistence": .328, "pdm": .284, "mal_prevalence": .01,
               "pop_growth_rate": .026, "pop_u5": 220_829, "mdp": .49, "household_size": 4.2,
               "modifiable_homes": .6455, "rdt_pos_24": 39_743},
}

param_labels = {
    "population": "Total Population", "households": "Number of Households",
    "bricks": "Proportion Brick Houses", "mud": "Proportion Mud Houses",
    "other_wall": "Other Wall Type Proportion", "subsistence": "Proportion Subsistence Farming",
    "pdm": "Proportion Receiving Parish Funds", "mal_prevalence": "Malaria Prevalence",
    "pop_growth_rate": "Population Growth Rate", "pop_u5": "Population Under 5",
    "mdp": "Multidimensional Poverty Index", "household_size": "Average Household Size",
    "modifiable_homes": "Proportion of Modifiable Homes", "rdt_pos_24": "RDT Positive Cases (2024)",
}