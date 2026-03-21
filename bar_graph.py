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
    print(args.data_src)

    data_frame = pandas.read_csv(args.data_src)
    data_frame = data_frame.sort_values('Data size').reset_index(drop = True)

    data_sizes = data_frame['Data size'].values
    abs_error_mean = data_frame['Mean'].values
    std_deviation = data_frame['Std. deviation'].values

    lower = abs_error_mean - std_deviation
    upper = abs_error_mean + std_deviation

    # generate plots
    plt.figure()

    plt.plot(
        data_sizes,
        abs_error_mean,
        marker = 'o',
        label = 'Abs. paklaidos vidurkis'
    )

    plt.fill_between(
        data_sizes,
        lower,
        upper,
        alpha = 0.1,
        label = 'Abs. paklaidos vidurkis ± std. nuokrypis'
    )

    plt.xlabel('Duomenų aibės dydis')
    plt.ylabel('Abs. paklaidos vidurkis')
    plt.title('Dirbtinio neuroninio tinklo rezultatai su sferos funkcijos duomenim\n(sigma + triuksmas)')
    plt.legend()
    
    plt.grid()
    plt.xscale('log')

    plt.show()