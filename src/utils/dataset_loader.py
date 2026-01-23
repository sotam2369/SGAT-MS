import argparse
import csv
import math
import os
import random
import re
from typing import Any, Dict
import shutil

import torch
from tqdm import tqdm
from data import SGATData


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for dataset loading and processing.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Input and output directories
    parser.add_argument(
        "-dir",
        "--directory",
        type=str,
        default="../dataset/original/train",
        help="Path to the input dataset directory",
    )
    parser.add_argument(
        "-out",
        "--output",
        type=str,
        default="../dataset/processed/train_dsize.pt",
        help="Path to save the processed dataset output file",
    )
    # Data splitting and filtering options
    parser.add_argument(
        "-s",
        "--split",
        type=int,
        default=1,
        help="Number of parts to split the dataset into",
    )
    parser.add_argument(
        "-ms",
        "--max-size",
        default=1,
        type=float,
        help="Maximum allowed file size in MB",
    )
    parser.add_argument(
        "-min",
        "--min-size",
        default=0,
        type=float,
        help="Minimum allowed file size in MB",
    )
    parser.add_argument(
        "-di",
        "--dataset-info",
        action="store_true",
        help="Print dataset information and exit",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the dataset files before processing",
    )
    parser.add_argument(
        "--tsplit",
        type=float,
        default=None,
        help="Fraction for test/train split (e.g., 0.2 means 20%% test data)",
    )
    parser.add_argument(
        "--ftype",
        type=int,
        default=0,
        help="Folder type: 0 for flat directory, 1 for nested directory structure",
    )
    parser.add_argument(
        "--copy-out",
        type=str,
        default=None,
        help="If set and source directory is unweighted, copy selected files into this directory under train/ and valid/",
    )
    return parser.parse_args()


def file_size_in_mb(file_path: str) -> float:
    """
    Calculate the size of a file in megabytes.

    Args:
        file_path (str): Path to the file.

    Returns:
        float: Size of the file in megabytes.
    """
    return os.stat(file_path).st_size / (1024 * 1024)


def get_problemset_name(rel_path: str, ftype: int) -> str | None:
    """
    Extract the problem set name from a relative file path based on folder type.

    Args:
        rel_path (str): Relative file path.
        ftype (int): Folder type (0 for flat, 1 for nested).

    Returns:
        str or None: Problem set name or None if undefined.
    """
    if ftype == 0:
        file = os.path.basename(rel_path)
        base = file.split(".")[0]
        for part in re.split(r'[-_]', base):
            if not part.isdigit():
                return part
        return base
    elif ftype == 1:
        return "/".join(rel_path.split("/")[:-1])
    return None


def get_file_list(
    directory: str,
    max_size: float,
    min_size: float,
    ftype: int,
    sort: bool,
    tsplit: float | None,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Retrieve a list of files from a directory filtered by size and type, optionally grouped by problem set.

    Args:
        directory (str): Root directory to search files.
        max_size (float): Maximum file size in MB.
        min_size (float): Minimum file size in MB.
        ftype (int): Folder type (0 for flat, 1 for nested).
        sort (bool): Whether to sort the files.
        tsplit (float or None): Test/train split fraction.

    Returns:
        tuple: (file_list, problemset) where file_list is list of filtered files,
               and problemset is a dict mapping problem set names to file lists.
    """
    problemset = {}
    file_list = []

    all_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), directory)
            all_files.append(rel_path)

    if sort:
        all_files.sort()

    for rel_path in all_files:
        file_path = os.path.join(directory, rel_path)
        size_mb = file_size_in_mb(file_path)
        if size_mb > max_size or size_mb <= min_size:
            continue
        if not (rel_path.endswith(".cnf") or rel_path.endswith(".wcnf")):
            continue

        file_list.append(rel_path)

        if tsplit is not None:
            problemset_name = get_problemset_name(rel_path, ftype)
            if problemset_name not in problemset:
                problemset[problemset_name] = [rel_path]
            else:
                problemset[problemset_name].append(rel_path)

    return file_list, problemset


def load_data(
    directory: str,
    filename: str,
) -> SGATData:
    """
    Load CNF data from a file and convert it to a data object.

    Args:
        directory (str): Directory containing the file.
        filename (str): Filename to load.

    Returns:
        SGATData: Loaded data object.
    """
    cnf_data = SGATData(os.path.join(directory, filename))
    cnf_data.quickLoad()
    return cnf_data.to_data()


SERIALIZATION_KEY = "_sgat_format"
SERIALIZATION_VERSION = "pyg_data_v1"


def _data_keys(data_obj) -> list[str]:
    """
    Retrieve the attribute keys for a PyG Data-like object.
    """
    keys_attr = getattr(data_obj, "keys", None)
    if callable(keys_attr):
        return list(keys_attr())
    if isinstance(keys_attr, (list, tuple)):
        return list(keys_attr)
    return [key for key in dir(data_obj) if not key.startswith("_")]


def serialize_data_object(data_obj) -> Dict[str, Any]:
    """
    Convert a PyTorch Geometric Data object into a dictionary composed only of
    built-in python containers and tensors so it can be safely loaded when
    torch.load defaults to weights_only=True.

    Args:
        data_obj: The PyG Data (or subclass) instance to serialize.

    Returns:
        dict: Serialized representation containing module/class metadata and tensor payload.
    """
    payload = {}
    for key in _data_keys(data_obj):
        try:
            payload[key] = data_obj[key]
        except (KeyError, TypeError, AttributeError):
            continue

    return {
        SERIALIZATION_KEY: SERIALIZATION_VERSION,
        "module": data_obj.__class__.__module__,
        "class": data_obj.__class__.__name__,
        "data": payload,
    }


def print_split_info(
    problemset_name: str,
    split: int,
    total: int,
) -> None:
    """
    Print information about data splitting for a problem set.

    Args:
        problemset_name (str): Name of the problem set.
        split (int): Number of training samples.
        total (int): Total number of samples.
    """
    train_pct = (split / total) * 100
    test_pct = ((total - split) / total) * 100
    print(f"{problemset_name:<30} Train: {split:<3} ({train_pct:.1f}%) | Test: {total - split} ({test_pct:.1f}%)")


def save_parts(
    output_file: str,
    data_list: list[str],
    splits: int,
    split_size: int,
    split_name: str,
) -> None:
    """
    Save data parts to separate files.

    Args:
        output_file (str): Base output filename.
        data_list (list): List of data files to process.
        splits (int): Number of parts to split into.
        split_size (int): Size of each split.
        split_name (str): Name of the split (e.g., "train" or "test").
    """
    for i in range(splits):
        part_data = data_list[i * split_size: min((i + 1) * split_size, len(data_list))]
        output_list = []
        for cnf_file in tqdm(part_data):
            pyg_data = load_data(args.directory, cnf_file)
            output_list.append(serialize_data_object(pyg_data))
        part_file = f"{output_file}.part{i+1}"
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(part_file), exist_ok=True)
        print(f"Saving...{i+1}/{splits}")
        torch.save(output_list, part_file)
        print("Saved")
        print(part_file)


def main() -> None:
    """
    Main function to process dataset files based on command-line arguments.
    """
    global args
    args = get_args()

    random.seed(0)

    listdir, problemset = get_file_list(args.directory, args.max_size, args.min_size, args.ftype, args.sort, args.tsplit)
    original_files = []
    for root, dirs, files in os.walk(args.directory):
        for f in files:
            if f.endswith(".cnf") or f.endswith(".wcnf"):
                original_files.append(os.path.relpath(os.path.join(root, f), args.directory))
    removed_files_count = len(original_files) - len(listdir)

    if args.tsplit is not None:
        train = []
        test = []
        fallback = []
        # Move any problemset with fewer than 3 files to fallback
        for problemset_name in list(problemset.keys()):
            if len(problemset[problemset_name]) < 3:
                fallback += problemset.pop(problemset_name)
                continue
        for problemset_name in problemset:
            random.shuffle(problemset[problemset_name])
            split = math.ceil((1 - args.tsplit) * len(problemset[problemset_name]))
            if split > 1 and split < 1 / args.tsplit:
                split -= 1
            train += problemset[problemset_name][:split]
            test += problemset[problemset_name][split:]
            total = len(problemset[problemset_name])
            print_split_info(problemset_name, split, total)
        if fallback:
            random.shuffle(fallback)
            split = math.ceil((1 - args.tsplit) * len(fallback))
            if split > 1 and split < 1 / args.tsplit:
                split -= 1
            train += fallback[:split]
            test += fallback[split:]
            print_split_info("fallback", split, len(fallback))

    # If requested, copy the actual selected files into separate train/valid folders
    # Only apply this when the source directory appears to be the unweighted set
    # and when a test/train split was requested.
    if args.copy_out is not None and args.tsplit is not None:
        src_dir = os.path.normpath(args.directory)
        copy_root = os.path.normpath(args.copy_out)
        train_dest = os.path.join(copy_root, "train/unsat")
        valid_dest = os.path.join(copy_root, "valid/unsat")
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(valid_dest, exist_ok=True)

        def copy_list(file_list, dest_dir):
            for rel in file_list:
                src = os.path.join(args.directory, rel)
                # preserve filename only (flat target)
                base = os.path.basename(rel)
                # If source is .wcnf, convert to .cnf by stripping weights
                if base.lower().endswith('.wcnf'):
                    dst_name = os.path.splitext(base)[0] + '.cnf'
                    dst = os.path.join(dest_dir, dst_name)
                    try:
                        with open(src, 'r') as fin, open(dst, 'w') as fout:
                            for line in fin:
                                # preserve comments and blank lines
                                if line.startswith('c') or line.strip() == '':
                                    fout.write(line)
                                    continue
                                # convert header 'p wcnf' to 'p cnf'
                                if line.lstrip().startswith('p wcnf'):
                                    parts = line.split()
                                    # parts: ['p', 'wcnf', num_vars, num_clauses, ...]
                                    if len(parts) >= 4:
                                        num_vars = parts[2]
                                        num_clauses = parts[3]
                                    else:
                                        num_vars = ''
                                        num_clauses = ''
                                    fout.write(f'p cnf {num_vars} {num_clauses}\n')
                                    continue

                                toks = line.strip().split()
                                if len(toks) == 0:
                                    continue
                                # Expect clause lines ending with '0'. First token is weight -> remove it.
                                if toks[-1] == '0' and len(toks) >= 2:
                                    # Remove leading weight if it's an integer
                                    clause_tokens = toks[1:] if len(toks) > 1 else toks
                                    # write clause (join tokens and ensure newline)
                                    fout.write(' '.join(clause_tokens) + '\n')
                                else:
                                    # Fallback: write the line as-is
                                    fout.write(line)
                    except Exception as e:
                        print(f"Warning: failed to convert {src} -> {dst}: {e}")
                else:
                    dst = os.path.join(dest_dir, base)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Warning: failed to copy {src} -> {dst}: {e}")

        print(f"Copying {len(train)} train files to {train_dest}")
        copy_list(train, train_dest)
        print(f"Copying {len(test)} valid files to {valid_dest}")
        copy_list(test, valid_dest)
    
    if args.dataset_info:
        dataset_info = [[], [], [], [], [], []]
        for cnf_file in tqdm(listdir):
            data = load_data(args.directory, cnf_file)
            dataset_info[0].append(cnf_file)
            dataset_info[1].append(data.num_nodes)
            dataset_info[2].append(data.num_edges)
            dataset_info[3].append(data.num_node_features if hasattr(data, 'num_node_features') else 0)
            dataset_info[4].append(0)
            dataset_info[5].append(0)

        with open(args.output, "w") as f:
            writer = csv.writer(f)
            writer.writerows(dataset_info)
        exit(0)

    print(f"Removed {removed_files_count} files")

    base_output_file = os.path.normpath(args.output)
    base_output_file = base_output_file.replace("dsize", str(len(listdir)))
    base_output_file = base_output_file.replace("maxsize", str(args.max_size))
    base_output_file = base_output_file.replace("minsize", str(args.min_size))

    if args.tsplit is not None:
        troutput_file = base_output_file.replace("type", "train")
        teoutput_file = base_output_file.replace("type", "test")
        train_splits = int(args.split * (1 - args.tsplit))
        test_splits = int(args.split * args.tsplit)
        train_split_size = int(math.ceil(len(train) / train_splits)) if train_splits > 0 else len(train)
        # Ensure parent directory exists before saving train parts
        os.makedirs(os.path.dirname(troutput_file), exist_ok=True)
        save_parts(troutput_file, train, train_splits, train_split_size, "train")
        test_split_size = int(math.ceil(len(test) / test_splits)) if test_splits > 0 else len(test)
        # Ensure parent directory exists before saving test parts
        os.makedirs(os.path.dirname(teoutput_file), exist_ok=True)
        save_parts(teoutput_file, test, test_splits, test_split_size, "test")
    else:
        listdir_split = int(math.ceil(len(listdir) / args.split))
        # Ensure parent directory exists before saving the output or its parts
        os.makedirs(os.path.dirname(base_output_file), exist_ok=True)
        for i in range(args.split):
            output_list = []
            for cnf_file in tqdm(listdir[i * listdir_split: min((i + 1) * listdir_split, len(listdir))]):
                pyg_data = load_data(args.directory, cnf_file)
                output_list.append(serialize_data_object(pyg_data))

            if args.split == 1:
                print("Saving...")
                torch.save(output_list, base_output_file)
                print("Saved")
                print(base_output_file)
            else:
                part_file = f"{base_output_file}.part{i+1}"
                # Ensure parent directory exists before saving each part
                os.makedirs(os.path.dirname(part_file), exist_ok=True)
                print(f"Saving...{i+1}/{args.split}")
                torch.save(output_list, part_file)
                print("Saved")
                print(part_file)


if __name__ == "__main__":
    main()
