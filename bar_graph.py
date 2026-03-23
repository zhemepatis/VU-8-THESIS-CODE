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
        "--graph-title",
        type = str,
        required = True,
        help = "Graph title"
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

    args = parser.parse_args()

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    # prepare labels
    methods = ["1nn", "2nn", "4nn", "8nn", "16nn", "32nn", "fnn"]
    data_sizes = ["1 tūkst.", "10 tūkst.", "100 tūkst.", "1 mln.", "10 mln."]
    colors = ["coral", "coral", "coral", "coral", "coral", "coral", "cornflowerblue"]
    
    x = np.arange(len(data_sizes))
    width = .8 / len(methods)

    fig, method_axis = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(methods):
        filtered = data_frame[
            (data_frame["Data function"] == args.benchmark_func) &
            (data_frame["Method"] == method) &
            (data_frame["Noise std. deviation"] == args.noise_std)
        ]

        offset = (i - len(methods) / 2 + 0.5) * width
        method_axis.bar(
            x + offset,
            filtered["Abs. error mean"].values,
            width = width,
            color = colors[i],
            edgecolor = "white",
            alpha = 1,
            label = method
        )

    # primary x-axis configuration   
    method_ticks = []
    method_labels = []
    for xi in x:
        for i, method in enumerate(methods):
            offset = (i - len(methods) / 2 + 0.5) * width
            method_ticks.append(xi + offset)
            method_labels.append(method)

    method_axis.set_xticks(method_ticks)
    method_axis.set_xticklabels(method_labels, fontsize = 10, rotation = 90)

    method_axis.set_xticks([tick for tick in method_ticks[::len(data_sizes)]] )

    method_axis.set_xticks(method_ticks)
    method_axis.set_xticklabels(methods * len(data_sizes), fontsize = 10, rotation = 90)
    method_axis.tick_params(axis = 'x', length = 0)

    # secondary x-axis configuration
    data_size_axis = method_axis.twiny()

    data_size_axis.xaxis.set_label_position('bottom')
    data_size_axis.xaxis.set_ticks_position('bottom')

    data_size_axis.set_xlim(method_axis.get_xlim())
    data_size_axis.spines['bottom'].set_position(('outward', 35))
    data_size_axis.set_xticks(x)
    data_size_axis.set_xticklabels(data_sizes, fontsize = 10)


    data_size_axis.set_title(f"{args.graph_title}\n(noise = 0, std = {args.noise_std})", fontsize = 14)
    data_size_axis.set_xlabel("Taškų kiekis duomenų aibėje", labelpad = 15, fontsize = 12)
    
    # y-axis configuration
    method_axis.set_ylabel("Abs. paklaidos vidurkis", labelpad = 15, fontsize = 12)
    method_axis.grid(axis = 'y', linestyle = '--', alpha = 0.5)

    plt.tight_layout()
    plt.savefig(args.graph_filename)