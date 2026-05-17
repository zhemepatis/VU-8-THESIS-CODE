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
        help = "Graph filename (will be suffixed with _1, _2, _3 before extension if no --data-function is specified)"
    )

    parser.add_argument(
        "--method",
        type = str,
        required = True,
        help = "Method name"
    )

    parser.add_argument(
        "--data-function",
        type = str,
        required = False,
        default = None,
        choices = ["sphere_func", "rosenbrock_func", "rastrigin_func"],
        help = "Data function to plot. If omitted, all three functions are plotted as separate files."
    )

    parser.add_argument(
        "--metric",
        type = str,
        required = True,
        choices = ["mean", "std"],
        help = "Metric to plot: 'mean' for absolute error mean, 'std' for absolute error std. deviation."
    )

    args = parser.parse_args()

    # resolve metric column and y-axis label
    metric_column_map = {
        "mean": "Absolute error mean",
        "std":  "Absolute error std. deviation",
    }
    metric_ylabel_map = {
        "mean": "Absoliučiosios paklaidos vidurkis",
        "std":  "Absoliučiosios paklaidos standartinis nuokrypis",
    }
    metric_column = metric_column_map[args.metric]
    metric_ylabel = metric_ylabel_map[args.metric]

    bar_color_map = {
        ("mean", True):  "steelblue",
        ("mean", False): "mediumseagreen",
        ("std",  True):  "#DBC2CF",
        ("std",  False): "#9FA2B2",
    }

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    # prepare labels
    all_data_functions = ["sphere_func", "rosenbrock_func", "rastrigin_func"]
    all_func_names     = ["Sferos funkcija", "Rozenbroko funkcija", "Rastrigino funkcija"]
    func_name_map = dict(zip(all_data_functions, all_func_names))

    if args.data_function is not None:
        functions_to_plot = [(args.data_function, func_name_map[args.data_function])]
    else:
        functions_to_plot = list(zip(all_data_functions, all_func_names))

    data_sizes = ["1 tūkst.", "10 tūkst.", "100 tūkst.", "1 mln."]
    if args.method != "fnn":
        data_sizes.append("10 mln.")

    noise_levels = [0, 5]

    x = np.arange(len(data_sizes))
    width = 0.25

    # split graph filename into stem and extension
    if "." in args.graph_filename:
        dot_index = args.graph_filename.rfind(".")
        name_stem = args.graph_filename[:dot_index]
        name_ext = args.graph_filename[dot_index:]
    else:
        name_stem = args.graph_filename
        name_ext = ""

    # plot one figure per function
    for idx, (func, func_name) in enumerate(functions_to_plot, start=1):
        fig, ax = plt.subplots(figsize=(5, 7))

        for i, noise_std_deviation in enumerate(noise_levels):
            bar_color = bar_color_map[(args.metric, noise_std_deviation == 0)]

            filtered = data_frame[
                (data_frame["Method"] == args.method) &
                (data_frame["Data function"] == func) &
                (data_frame["Noise std. deviation"] == noise_std_deviation) &
                ~((args.method == "fnn") & (data_frame["Data size"] == 10_000_000))
            ]

            offset = (i - 1) * width
            ax.bar(
                x + offset,
                filtered[metric_column].values,
                width = width,
                color = bar_color,
                edgecolor = "white",
                label = f"μ = {0.0}, σ = {noise_std_deviation:.1f}",
                alpha = .75
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data_sizes)
        ax.set_xlabel("Taškų skaičius", labelpad = 10, fontsize = 12)
        ax.set_ylabel(metric_ylabel, labelpad = 10, fontsize = 12)
        ax.grid(True, axis='y')
        ax.legend()

        plt.tight_layout()
        # use a suffix only when plotting all functions
        if args.data_function is None:
            output_path = f"{name_stem}_{idx}{name_ext}"
        else:
            output_path = args.graph_filename
        plt.savefig(output_path)
        plt.close(fig)