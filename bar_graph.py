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
    data_frame = data_frame.sort_values(by = ["Method", "Data function", "Noise std. deviation", "Data size"]).reset_index(drop = True)
    
    print(data_frame[data_frame["Method"] == "fnn"])
