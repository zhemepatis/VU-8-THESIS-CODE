import numpy as np
import pandas
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(description = "Bar graph generator args parser")

    parser.add_argument(
        "--data-src-filename",
        type = str,
        required = True,
        help = "Source data filename"
    )

    parser.add_argument(
        "--graph-filename",
        type = str,
        required = True,
        help = "Graph filename"
    )

    parser.add_argument(
        "--method",
        type = str,
        required = True,
        help = "Method name"
    )

    parser.add_argument(
        "--noise-std",
        type = float,
        required = True,
        help = "Noise std. deviation to include in the graph"
    )

    parser.add_argument(
        "--error-type",
        type = str,
        required = True,
        choices = ["absolute", "normalized", "relative"],
        help = "Error type to display: absolute, normalized, or relative"
    )

    parser.add_argument(
        "--stat",
        type = str,
        required = True,
        choices = ["std", "mean"],
        help = "Statistic to display: std (standard deviation) or mean"
    )

    parser.add_argument(
        "--data-function",
        type = str,
        required = True,
        choices = ["sphere_func", "rosenbrock_func", "rastrigin_func"],
        help = "Data function to use: sphere_func, rosenbrock_func, or rastrigin_func"
    )

    args = parser.parse_args()

    # build column name from error type and stat arguments
    error_type_label = {
        "absolute":   "Absolute error",
        "normalized": "Normalized error",
        "relative":   "Relative error",
    }[args.error_type]

    stat_label = {
        "std":  "std. deviation",
        "mean": "mean",
    }[args.stat]

    column_name = f"{error_type_label} {stat_label}"

    y_axis_label = {
        ("absolute",   "std"):  "Absoliučiosios paklaidos standartinis nuokrypis",
        ("absolute",   "mean"): "Absoliučiosios paklaidos vidurkis",
        ("normalized", "std"):  "Normalizuotos paklaidos standartinis nuokrypis",
        ("normalized", "mean"): "Normalizuotos paklaidos vidurkis",
        ("relative",   "std"):  "Santykinės paklaidos standartinis nuokrypis",
        ("relative",   "mean"): "Santykinės paklaidos vidurkis",
    }[(args.error_type, args.stat)]

    bar_color = {
        ("mean", True):  "steelblue",
        ("mean", False): "mediumseagreen",
        ("std",  True):  "#DBC2CF",
        ("std",  False): "#9FA2B2",
    }[(args.stat, args.noise_std == 0)]

    func_name = {
        "sphere_func":     "Sferos funkcija",
        "rosenbrock_func": "Rozenbroko funkcija",
        "rastrigin_func":  "Rastrigino funkcija",
    }[args.data_function]

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    # prepare labels
    data_sizes = ["1 tūkst.", "10 tūkst.", "100 tūkst.", "1 mln."]
    if args.method != "fnn":
        data_sizes.append("10 mln.")

    x = np.arange(len(data_sizes))
    width = 0.3

    # plot
    fig, ax = plt.subplots(figsize=(5, 7))

    filtered = data_frame[
        (data_frame["Method"] == args.method) &
        (data_frame["Data function"] == args.data_function) &
        (data_frame["Noise std. deviation"] == args.noise_std) &
        ~((args.method == "fnn") & (data_frame["Data size"] == 10_000_000))
    ]

    ax.bar(
        x,
        filtered[column_name].values,
        width = width,
        color = bar_color,
        edgecolor = "white",
        label = f"μ = 0.0, σ = {args.noise_std:.1f}",
        alpha = .75
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data_sizes)
    ax.set_xlabel("Taškų skaičius", labelpad = 10, fontsize = 12)
    ax.set_ylabel(y_axis_label, labelpad = 10, fontsize = 12)
    ax.grid(True, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.graph_filename)
    plt.close(fig)