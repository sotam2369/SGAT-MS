import pandas as pd
import argparse
import os

def combine_solver_results(files, output_file):
    combined = None
    for f in files:
        suffix = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        df = df.add_suffix(f'_{suffix}')
        df = df.rename(columns={f'problem_{suffix}': 'problem'})
        if combined is None:
            combined = df
        else:
            combined = pd.merge(combined, df, on='problem')
    sum_row = combined.select_dtypes(include='number').sum()
    sum_row['problem'] = 'TOTAL'
    combined = pd.concat([combined, pd.DataFrame([sum_row])], ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple solver result CSV files by problem name.")
    parser.add_argument("--output", help="Path to the output CSV file")
    parser.add_argument("files", nargs='+', help="Paths to the CSV files to combine")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combine_solver_results(args.files, args.output)
