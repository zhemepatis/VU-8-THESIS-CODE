import numpy as np
import pandas
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(description = "Graph generator args parser")

    parser.add_argument(
        "--data-src",
        type = str,
        required = True,
        help = "Data file name"
    )

    parser.add_argument(
        "--graph-name",
        type = str,
        required = True,
        help = "Graph picture file name"
    )

    args = parser.parse_args()

    # get data
    data_frame = pandas.read_csv(args.data_src)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(by=["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)

    methods = ["1nn", "2nn", "4nn", "8nn", "16nn", "32nn", "fnn"]
    data_sizes = ["1 tūkst.", "10 tūkst.", "100 tūkst.", "1 mln.", "10 mln."]
    colors = ['cornflowerblue', 'lightsalmon', 'mediumaquamarine', 'plum', 'khaki', 'lightcoral', 'powderblue']

    x = np.arange(len(data_sizes))
    width = .5 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        filtered = data_frame[(data_frame["Method"] == method) & (data_frame["Noise std. deviation"] == 0)]
        
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            filtered["Abs. error mean"].values,
            width = width,
            color = colors[i],
            alpha = 0.8,
            label = method
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(data_sizes)
    ax.legend()

    ax.set_title("Modelių rezultatai su sferos funkcijos generuotais duomenim")
    ax.set_xlabel("Taškų kiekis duomenų aibėje")
    ax.set_ylabel("Abs. paklaidos vidurkis")
    
    ax.legend()
    ax.grid(axis = "y", linestyle = "--", alpha = 0.5)

    plt.tight_layout()
    plt.show()