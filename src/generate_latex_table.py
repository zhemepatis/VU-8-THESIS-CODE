import pandas
import argparse


def format_data_size(n):
    """Format number with spaces as thousands separator (Lithuanian style)."""
    return f"{n:,}".replace(",", " ")


def format_value(val, threshold=1e-6):
    """Format a float value, using <10^{-6} notation for very small numbers."""
    if val < threshold:
        return r"$<10^{-6}$"
    return f"{val:.6f}"


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(description="LaTeX table generator args parser")

    parser.add_argument(
        "--data-src-filename",
        type=str,
        required=True,
        help="Source data filename"
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Gaussian noise std"
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Filter by method name"
    )

    parser.add_argument(
        "--data-function",
        type=str,
        default=None,
        help="Filter by data function name"
    )

    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Custom caption for the table (LaTeX string)"
    )

    args = parser.parse_args()

    # get data
    data_frame = pandas.read_csv(args.data_src_filename)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.sort_values(
        by=["Method", "Data function", "Noise std. deviation", "Data size"]
    ).reset_index(drop=True)

    # apply filters
    if args.method is not None:
        data_frame = data_frame[data_frame["Method"] == args.method]

    if args.data_function is not None:
        data_frame = data_frame[data_frame["Data function"] == args.data_function]

    if args.noise_std is not None:
        data_frame = data_frame[data_frame["Noise std. deviation"] == args.noise_std]

    # group by data size and compute stats
    grouped = (
        data_frame
        .groupby("Data size")["Error"]   # adjust "Error" column name if needed
        .agg(Min="min", Max="max", Mean="mean", Std="std")
        .reset_index()
        .sort_values("Data size")
    )

    # build caption
    if args.caption:
        caption = args.caption
    else:
        method_str = args.method if args.method else "?"
        noise_str = args.noise_std if args.noise_std is not None else "0"
        caption = rf"$k$ artimiausių kaimynų metodo rezultatai ($k = 1$, $\mu = 0$, $\sigma = {noise_str}$)"

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

    for _, row in grouped.iterrows():
        size_str = format_data_size(int(row["Data size"]))
        min_str  = format_value(row["Min"])
        max_str  = format_value(row["Max"])
        mean_str = format_value(row["Mean"])
        std_str  = format_value(row["Std"])

        lines.append(
            f"        {size_str:<12} & {min_str} & {max_str} & {mean_str} & {std_str} \\\\ \\hline"
        )

    lines.append(r"    \end{tabular}")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(r"\end{table}")

    print("\n".join(lines))