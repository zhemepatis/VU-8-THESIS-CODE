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
        "--benchmark-func",
        type = str,
        default = None,
        help = "Benchmark function name"
    )

    parser.add_argument(
        "--noise-std",
        type = float,
        default = None,
        help = "Gaussian noise std"
    )

    parser.add_argument(
        "--metric",
        type = str,
        choices = ["mean", "std"],
        default = "mean",
        help = "Metric to plot: 'mean' for Absolute error mean, 'std' for Absolute error std. deviation"
    )

    args = parser.parse_args()

    metric_column = "Absolute error mean" if args.metric == "mean" else "Absolute error std. deviation"
    metric_label = "Absoliučiosios paklaidos vidurkis" if args.metric == "mean" else "Absoliučiosios paklaidos standartinis nuokrypis"

    color_map = {
        ("mean", True):  "coral",
        ("mean", False): "#BBA0CA",
        ("std",  True):  "#DBC2CF",
        ("std",  False): "#9FA2B2",
    }
    color = color_map.get((args.metric, args.noise_std == 0), "gray")

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    # prepare labels
    methods = ["1nn", "2nn", "4nn", "8nn", "16nn", "32nn"]
    method_names = ["1NN", "2NN", "4NN", "8NN", "16NN", "32NN"]

    data_sizes = ["1 tūkst. taškų", "10 tūkst. taškų", "100 tūkst. taškų", "1 mln. taškų", "10 mln. taškų"]

    x = np.arange(len(data_sizes))
    width = .8 / len(methods)

    fig, method_axis = plt.subplots(figsize=(12, 7))

    for i, method in enumerate(methods):
        filtered = data_frame[
            (data_frame["Data function"] == args.benchmark_func) &
            (data_frame["Method"] == method) &
            (data_frame["Noise std. deviation"] == args.noise_std)
        ]

        offset = (i - len(methods) / 2 + 0.5) * width
        method_axis.bar(
            x + offset,
            filtered[metric_column].values,
            width = width,
            color = color,
            edgecolor = "white",
            alpha = 1,
            label = method
        )

    # primary x-axis configuration   
    method_ticks = []
    for xi in x:
        for i, method in enumerate(method_names):
            offset = (i - len(method_names) / 2 + 0.5) * width
            method_ticks.append(xi + offset)

    method_axis.set_xticks(method_ticks)
    method_axis.set_xticklabels(method_names * len(data_sizes), fontsize = 10, rotation = 90)
    method_axis.tick_params(axis = 'x', length = 0)

    # secondary x-axis configuration
    data_size_axis = method_axis.twiny()

    data_size_axis.xaxis.set_label_position('bottom')
    data_size_axis.xaxis.set_ticks_position('bottom')

    data_size_axis.set_xlim(method_axis.get_xlim())
    data_size_axis.spines['bottom'].set_position(('outward', 35))
    data_size_axis.set_xticks(x)
    data_size_axis.set_xticklabels(data_sizes, fontsize = 10)

    data_size_axis.set_xlabel("Taškų kiekis duomenų aibėje", labelpad = 15, fontsize = 12)
    
    # y-axis configuration
    method_axis.set_ylabel(metric_label, labelpad = 15, fontsize = 12)
    method_axis.grid(axis = 'y', linestyle = '--', alpha = 0.5)

    plt.tight_layout()
    plt.savefig(args.graph_filename)