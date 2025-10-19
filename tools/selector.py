import os
import shutil
import argparse

def filter_maxsat_instances(in_dir, out_dir):
    listdir = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if not f.endswith("cnf"):
                continue
            file_path = os.path.join(root, f)
            has_hard = False
            with open(file_path, 'r') as file:
                for line in file:
                    if '"nhards"' in line:
                        try:
                            hard_count = int(line.split(":")[1].split(",")[0].strip())
                        except (IndexError, ValueError):
                            hard_count = 0
                        if hard_count > 0:
                            has_hard = True
                            break

            rel_path = os.path.relpath(file_path, in_dir)
            if has_hard:
                os.remove(os.path.join(in_dir, rel_path))
                continue

            listdir.append(rel_path)

    for file in listdir:
        os.makedirs(os.path.dirname(os.path.join(out_dir, file)), exist_ok=True)
        shutil.copyfile(os.path.join(in_dir, file), os.path.join(out_dir, file))

def main():
    parser = argparse.ArgumentParser(description="Filter MaxSAT instances based on header constraints.")
    parser.add_argument("in_dir", type=str, help="Input directory with original dataset.")
    parser.add_argument("out_dir", type=str, help="Output directory for filtered dataset.")
    args = parser.parse_args()
    filter_maxsat_instances(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()
