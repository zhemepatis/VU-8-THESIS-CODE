import pandas
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
    print(data_frame)