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

    args = parser.parse_args()

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    # prepare labels
    data_functions = ["sphere_func", "rosenbrock_func", "rastrigin_func"]
    func_names = ["Sferos funkcija", "Rozenbroko funkcija", "Rastrigino funkcija"]
    
    data_sizes = ["1 tūkst.", "10 tūkst.", "100 tūkst.", "1 mln."]
    if args.method != "fnn":
        data_sizes.append("10 mln.")

    noise_levels = [0, 5]
    colors = ["steelblue", "mediumseagreen"]

    # plot graphs
    x = np.arange(len(data_sizes))
    width = 0.25

    fig, axes = plt.subplots(1, len(data_functions), figsize=(5 * len(data_functions), 7))

    for ax, func, func_name in zip(axes, data_functions, func_names):
        for i, noise_std_deviation in enumerate(noise_levels):
            filtered = data_frame[
                (data_frame["Method"] == args.method) &
                (data_frame["Data function"] == func) &
                (data_frame["Noise std. deviation"] == noise_std_deviation) &
                ~((args.method == "fnn") & (data_frame["Data size"] == 10_000_000))
            ]

            offset = (i - 1) * width
            ax.bar(
                x + offset,
                filtered["Abs. error mean"].values,
                width = width,
                color = colors[i],
                edgecolor = "white",
                label = f"μ = {0.0}, σ = {noise_std_deviation:.1f}",
                alpha = .75
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data_sizes)
        ax.set_title(func_name)
        ax.set_xlabel("Taškų skaičius", labelpad = 10, fontsize = 12)
        ax.grid(True, axis='y')

    axes[0].set_ylabel("Abs. paklaidos vidurkis", labelpad = 10, fontsize = 12)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.graph_filename)