import pandas
import argparse

def format_data_size(n):
    return f"{n:,}".replace(",", " ")


def format_value(val, threshold=1e-6):
    if val < threshold:
        return r"$<10^{-6}$"
    
    return f"{val:.6f}"


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(description="LaTeX table generator args parser")

    parser.add_argument(
        "--data-src-filename",
        type = str,
        required = True,
        help = "Source data filename"
    )

    parser.add_argument(
        "--table-filename",
        type = str,
        required = True,
        help = "Table filename"
    )

    parser.add_argument(
        "--method",
        type = str,
        required = True,
        help = "Filter by method name"
    )

    parser.add_argument(
        "--benchmark-func",
        type = str,
        required = True,
        help = "Filter by data function name"
    )

    parser.add_argument(
        "--noise-std",
        type = float,
        required = True,
        help = "Gaussian noise std"
    )

    parser.add_argument(
        "--caption",
        type = str,
        required = True,
        help = "Caption for the table"
    )

    args = parser.parse_args()

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(
        by = ["Method", "Data function", "Noise std. deviation", "Data size"]
    ).reset_index(drop = True)

    # apply filters
    filtered = data_frame[
        (data_frame["Method"] == args.method) &
        (data_frame["Data function"] == args.benchmark_func) &
        (data_frame["Noise std. deviation"] == args.noise_std) &
        ~((args.method == "fnn") & (data_frame["Data size"] == 10_000_000))
    ]

    # generate LaTeX
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tabular}{|c|c|c|c|c|}")
    lines.append(r"        \hline")
    lines.append(
        r"        \textbf{Taškų kiekis} & \textbf{Min} & \textbf{Max} & "
        r"\textbf{Vidurkis} & \textbf{Std. nuokrypis} \\ \hline"
    )

    for _, row in filtered.iterrows():
        size_str = format_data_size(int(row["Data size"]))
        min_str  = format_value(row["Abs. error min"])
        max_str  = format_value(row["Abs. error max"])
        mean_str = format_value(row["Abs. error mean"])
        std_str  = format_value(row["Abs. error std. deviation"])

        lines.append(
            f"        {size_str:<12} & {min_str} & {max_str} & {mean_str} & {std_str} \\\\ \\hline"
        )

    lines.append(r"    \end{tabular}")
    lines.append(f"    \\caption{{{args.caption}}}")
    lines.append(r"\end{table}")

    # print to .tex file
    output = "\n".join(lines)

    with open(args.table_filename, "w", encoding="utf-8") as file:
        file.write(output)

    print(f"Table written to {args.table_filename}")