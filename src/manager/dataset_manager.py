import importlib
import inspect
import pickle
from typing import Any, List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

SERIALIZATION_KEY = "_sgat_format"
SERIALIZATION_VERSION = "pyg_data_v1"


def _allow_torch_safe_globals() -> None:
    """
    Allowlist PyG data classes for torch.load in case we fall back to
    weights_only=False on older dataset archives.
    """
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return
    try:
        from torch_geometric.data.data import DataEdgeAttr  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        DataEdgeAttr = None
    if DataEdgeAttr is not None:
        try:
            add_safe_globals([DataEdgeAttr])
        except TypeError:
            # PyTorch may have already registered the class; ignore.
            pass


def _torch_load_compat(path: str, **kwargs: Any) -> Any:
    """
    Load a torch object while remaining compatible with both the new
    weights_only=True default and legacy pickled datasets.
    """
    try:
        return torch.load(path, **kwargs)
    except (pickle.UnpicklingError, RuntimeError) as err:
        message = str(err)
        if not isinstance(err, pickle.UnpicklingError) and "Weights only load failed" not in message and "unsupported" not in message.lower():
            raise

        load_kwargs = dict(kwargs)
        signature = inspect.signature(torch.load)
        if "weights_only" in signature.parameters:
            load_kwargs["weights_only"] = False

        _allow_torch_safe_globals()
        return torch.load(path, **load_kwargs)


def _deserialize_entry(entry: Any) -> Any:
    """
    Convert serialized dictionaries produced by dataset_loader back into
    PyG Data objects. Legacy Data objects are returned untouched.
    """
    if isinstance(entry, Data):
        return entry

    if isinstance(entry, dict) and entry.get(SERIALIZATION_KEY) == SERIALIZATION_VERSION:
        module_name = entry.get("module")
        class_name = entry.get("class")
        payload = entry.get("data", {})

        data_cls = Data
        if module_name and class_name:
            try:
                module = importlib.import_module(module_name)
                data_cls = getattr(module, class_name, Data)
            except (ImportError, AttributeError):
                data_cls = Data

        return data_cls(**payload)

    return entry


def _deserialize_sequence(sequence: Any) -> List[Any]:
    """
    Apply entry deserialization across a loaded dataset component.
    """
    if isinstance(sequence, list):
        return [_deserialize_entry(item) for item in sequence]
    return [_deserialize_entry(sequence)]

class SGATDataset:
    def load_from_pickle(
        self,
        train_path: str,
        test_path: str,
        train_split: int,
        test_split: int,
        batch_size: int,
        test_batch_size: int = 1,
    ) -> None:
        """
        Load training and testing datasets from pickle files.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.
            train_split (int): Number of parts for training split.
            test_split (int): Number of parts for testing split.
            batch_size (int): Batch size for training data loader.
            test_batch_size (int): Batch size for test data loader.
            random_init (bool): Whether to apply random initialization to data.
        """
        is_same_dataset = (train_path == test_path)
        offset = is_same_dataset * train_split

        self.train = DataLoader(
            self.load_dataset(train_path, train_split),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        self.test = DataLoader(
            self.load_dataset(test_path, test_split, offset=offset),
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=True
        )

    def load_dataset(
        self,
        path: str,
        split: int,
        offset: int = 0
    ):
        """
        Load a dataset from a single or multiple pickle files.

        Args:
            path (str): Base path to the dataset.
            split (int): Number of splits (0 = single file).
            offset (int): Offset for part index.
            random_init (bool): Whether to apply random initialization to data.

        Returns:
            List of converted data objects.
        """
        if split == 0:
            data_list = _torch_load_compat(path)
            return _deserialize_sequence(data_list)

        dataset = []
        for i in range(split):
            file_path = f"{path}.part{i + 1 + offset}"
            part_data = _torch_load_compat(file_path)
            dataset.extend(_deserialize_sequence(part_data))

        return dataset
