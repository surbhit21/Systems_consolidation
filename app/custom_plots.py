import csv
import json
import os

from LoadNPLot2 import compare_freezing_on_day, plot_goto_fig2c_crossover_latency


def write_stats(stats, output_basename):
    """Write full statistics to JSON and pairwise p-values to CSV."""
    output_directory = os.path.dirname(output_basename)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open("{}_stats.json".format(output_basename), "w") as stats_file:
        json.dump(stats, stats_file, indent=4)

    comparisons = stats.get("comparisons", [])
    csv_path = "{}_pvalues.csv".format(output_basename)
    with open(csv_path, "w", newline="") as pvalue_file:
        fieldnames = [
            "group1",
            "group2",
            "index1",
            "index2",
            "n1",
            "n2",
            "test",
            "statistic",
            "pvalue_raw",
            "pvalue_corrected",
            "correction",
            "significant",
            "stars",
        ]
        writer = csv.DictWriter(pvalue_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)

    print("\nStatistics: {}".format(output_basename))
    for comparison in comparisons:
        print(
            "  {} (n={}) vs {} (n={}): U={:.3g}, p={:.4g}, "
            "corrected p={:.4g} ({}) {}".format(
                comparison["group1"],
                comparison["n1"],
                comparison["group2"],
                comparison["n2"],
                comparison["statistic"],
                comparison["pvalue_raw"],
                comparison["pvalue_corrected"],
                comparison["correction"],
                comparison["stars"],
            )
        )

# Fig 5 C
comparison_pairs=[(0, 1), (1, 2), (0, 2)]
plot_data, stats = compare_freezing_on_day(
    [
        "CNT_fast_drift_with_limited11_IP_lowI",
        "HPCLTPErase[0]_fast_drift_with_limited11_IP_lowI",
        "HPCLTPErase[1]_fast_drift_with_limited11_IP_lowI",
    ],
     xtick_labels=[
        "Control",
        "L",
        "OF1",
    ],
    x_group_labels=[
        ("LTP erasure", 1, 2),
    ],
    day=2,
    labels=["Control", "LTP Erasure L", "OF1"],
    n_presentations=10,
    comparison_pairs=comparison_pairs,
    title = "",
    bar_colors = ["#d25d003f", "#d25d003f", "#d25d003f"],
    fname="./plots/plasticity_blocking/HPC_blocking_freezing_day{0}_comparison_wo_LatePlasticity".format(2),
)
write_stats(stats, "./plots/plasticity_blocking/HPC_blocking_freezing_day2_comparison_wo_LatePlasticity")

plot_data, stats = plot_goto_fig2c_crossover_latency(
    file_names=[
        "Fig2C_and_Fig3A_SN_CALI_D2Recall.csv",
        "Fig2C_CALI_D2Recall.csv",
        "Fig3A_CALI_D2recall.csv",
    ],
    condition_order=[
        "Control",
        "IA",
        "OF1",
    ],
    control_index = 0,
    xtick_labels=[
        "Control",
        "IA",
        "OF1",
    ],
    x_group_labels=[
        ("CALI", 1, 2),
    ],
    title="",
    comparison_pairs=comparison_pairs,
    ylabel=r"$\Delta$ Crossover Latency (s)",
    bar_colors=["#d25d003f", "#d25d003f","#d25d003f"],
    fname="./plots/plasticity_blocking/Goto_Fig2C3A_crossover_latency"
)
write_stats(stats, "./plots/plasticity_blocking/Goto_Fig2C3A_crossover_latency_wo_LatePlasticity")



# Fig 5 D
plot_data, stats = compare_freezing_on_day(
    [
        "CNT_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[0]_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[1]_fast_drift_with_limited11_IP_lowI",
    ],
     xtick_labels=[
        "Control",
        "L",
        "OF1",
    ],
    x_group_labels=[
        ("LTP erasure", 1, 2),
    ],
    comparison_pairs=comparison_pairs,
    day=2,
    labels=["Control", "L", "OF1"],
    n_presentations=10,
    title = "",
    bar_colors = ["#90abe618", "#90abe618", "#90abe618"],
    fname="./plots/plasticity_blocking/ACC_blocking_freezing_day{0}_comparison_wo_LatePlasticity".format(2),
)
write_stats(stats, "./plots/plasticity_blocking/ACC_blocking_freezing_day2_comparison_wo_LatePlasticity")

plot_data, stats = plot_goto_fig2c_crossover_latency(
    file_names=[
        "Fig5B_and_Fig5C_CNT_D2Recall.csv",
        "Fig5B_CALI_D2Recall.csv",
        "Fig5C_CALI_D2Recall.csv",
    ],
    condition_order=[
        "Control",
        "IA",
        "OF1",
    ],
    control_index = 0,
    xtick_labels=[
        "Control",
        "IA",
        "OF1",
    ],
    x_group_labels=[
        ("CALI", 1, 2),
    ],
    title="",
    comparison_pairs=comparison_pairs,
    ylabel=r"$\Delta$ Crossover Latency (s)",
    bar_colors=["#90abe618", "#90abe618","#90abe618"],
    fname="./plots/plasticity_blocking/Goto_Fig5B5C_crossover_latency"
)
write_stats(stats, "./plots/plasticity_blocking/Goto_Fig5B5C_crossover_latency_wo_LatePlasticity")

comparison_pairs=[(0, 1)]
# Fig 5 E
plot_data, stats = compare_freezing_on_day(
    [
        "CNT_fast_drift_with_limited11_IP_lowI",
        "HPCLTPErase[2]_fast_drift_with_limited11_IP_lowI",
    ],
     xtick_labels=[
        "Control",
        "LTP Erasure",
    ],
    comparison_pairs=comparison_pairs,
    day=3,
    labels=["Control", "LTP Erasure"],
    n_presentations=10,
    title = "",
    bar_colors = ["#d25d003f", "#d25d003f"],
    fname="./plots/plasticity_blocking/HPC_blocking_freezing_day{0}_comparison_wo_LatePlasticity".format(3),
)
write_stats(stats, "./plots/plasticity_blocking/HPC_blocking_freezing_day3_comparison_wo_LatePlasticity")

plot_data, stats = plot_goto_fig2c_crossover_latency(
    file_names=[
        "Fig_3B_CNT.csv",
        "Fig3B_D2CALI_D3Recall.csv"
    ],
    condition_order=[
        "Control",
        "CALI (Day 2)",
    ],
    control_index = 0,
    xtick_labels=[
        "Control",
        "CALI (Day 2)",
    ],
    title="",
    comparison_pairs=comparison_pairs,
    ylabel=r"$\Delta$ Crossover Latency (s)",
    bar_colors=["#d25d003f", "#d25d003f"],
    fname="./plots/plasticity_blocking/Goto_Fig3B_crossover_latency"
)
write_stats(stats, "./plots/plasticity_blocking/Goto_Fig3B_crossover_latency")


# Fig 5 F
plot_data, stats = compare_freezing_on_day(
    [
        # "CNT_fast_drift_with_limited11_IP_lowI",
        # "ACCLTPErase[2]_fast_drift_with_limited11_IP_lowI",
        "CNT_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[2]_fast_drift_with_limited11_IP_lowI",
    ],
    xtick_labels=[
        "Control",
        "LTP Erasure",
    ],
    comparison_pairs=comparison_pairs,
    day=3,
    labels=["Control", "LTP Erasure (Day 2)"],
    n_presentations=10,
    title = "",
    bar_colors = ["#90abe618", "#90abe618"],
    fname="./plots/plasticity_blocking/ACC_blocking_freezing_day{0}_comparison_wo_LatePlasticity".format(3),
)
write_stats(stats, "./plots/plasticity_blocking/ACC_blocking_freezing_day3_comparison_wo_LatePlasticity")

plot_data, stats = plot_goto_fig2c_crossover_latency(
    file_names=[
        "Fig5D_CNT_D3Recall.csv",
        "Fig5D_CALI_D3Recall.csv",
    ],
    condition_order=[
        "Control",
        "CALI (Day 2)",
    ],
    control_index = 0,
    xtick_labels=[
        "Control",
        "CALI (Day 2)",
    ],
    title="",
    comparison_pairs=comparison_pairs,
    ylabel=r"$\Delta$ Crossover Latency (s)",
    bar_colors=["#90abe618", "#90abe618"],
    fname="./plots/plasticity_blocking/Goto_Fig5D_crossover_latency"
)
write_stats(stats, "./plots/plasticity_blocking/Goto_Fig5D_crossover_latency_wo_LatePlasticity")


comparison_pairs=[(0,3), (1,3),(2,3)]
# supp_liu_2025 D
plot_data, stats = plot_goto_fig2c_crossover_latency(
    file_names=[
        "Liu_2025_Fig3B_CNT_D4Recall.csv",
        "Liu_2025_Fig3B_D2CALI_D4Recall.csv",
        "Liu_2025_Fig3B_D3CALI_D4Recall.csv",
        "Liu_2025_Fig3B_D2D3CALI_D4Recall.csv"
    ],
    condition_order=[
        "Control",
        "CALI (Day 2)",
        "CALI (Day 3)",
        "CALI (Day 2 & 3)",
    ],
    control_index = 3,
    xtick_labels=[
        "Control",
        "Day 2",
        "Day 3",
        "Day 2 & 3",
    ],
    x_group_labels=[
        ("CALI", 1, 3),
    ],
    comparison_pairs=comparison_pairs,
    title="",
    ylabel=r"$\Delta$ Crossover Latency (s)",
    bar_colors=["#90abe618", "#90abe618","#90abe618","#90abe618"],
    fname="./plots/plasticity_blocking/Liu2025_Fig3B_crossover_latency"
)
write_stats(stats, "./plots/plasticity_blocking/Liu2025_Fig3B_crossover_latency_wo_LatePlasticity")

# supp_liu_2025 E
plot_data, stats = compare_freezing_on_day(
    [
        "CNT_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[2]_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[3]_fast_drift_with_limited11_IP_lowI",
        "ACCLTPErase[2, 3]_fast_drift_with_limited11_IP_lowI"
    ],
     xtick_labels=[
        "Control",
        "Day 2",
        "Day 3",
        "Day 2 & 3",
    ],
    control_index=3,
    x_group_labels=[
        ("LTP Erasure", 1,  3),
    ],
    day=4,
    comparison_pairs=comparison_pairs,
    labels=["Control", "LTP Erasure Day 2", "LTP Erasure Day 3", "LTP Erasure Day 2 & 3"],
    n_presentations=10,
    bar_colors = ["#90abe618", "#90abe618", "#90abe618", "#90abe618"],
    title = "",
    fname="./plots/plasticity_blocking/ACC_blocking_freezing_day{0}_comparison_wo_LatePlasticity".format(4),
)
write_stats(stats, "./plots/plasticity_blocking/ACC_blocking_freezing_day4_comparison_wo_LatePlasticity")
