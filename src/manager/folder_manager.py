from pathlib import Path
import shutil
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Union
import pandas as pd


class SGATFolder:
    def __init__(
        self,
        directory: str = "../plots/",
        folder_path: Optional[str] = None,
        b_weights: Optional[List[float]] = None
    ):
        """
        Initialize SGATFolder for managing training outputs.

        Args:
            directory (str): Base directory to create folders in.
            folder_path (str, optional): Specific folder path to use.
            minimize (List[float], optional): List indicating whether to minimize (1) or maximize (-1) each metric.
            b_weights (List[float], optional): Weights for each metric in comparison.
        """
        if isinstance(b_weights, torch.Tensor):
            weights = b_weights.detach().cpu().tolist()
        else:
            weights = list(b_weights) if b_weights is not None else []
        if not weights:
            weights = [1.0]
        else:
            weights = [weights[0]]
        self.b_weights = torch.tensor(weights, dtype=torch.float32)
        self.minimize = torch.tensor([1 if b > 0 else -1 for b in self.b_weights], dtype=torch.float32)
        self.best_model_val: Optional[torch.Tensor] = None
        self.saved_epoch: int = -1

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.folder_path = Path(folder_path) if folder_path else self._create_unique_folder(directory)

    def _create_unique_folder(
        self,
        directory: Path
    ) -> Path:
        for i in range(1, 10000):
            path = directory / f"train_{i}"
            if not path.exists() or len(list(path.iterdir())) <= 2:
                path.mkdir(parents=True, exist_ok=True)
                self.folder_id = i
                return path
        raise RuntimeError("Failed to create a unique training directory.")

    def save_plot(
        self,
        data_list: List[np.ndarray],
        name_list: List[str],
        file_name: str,
        add_border: Optional[Callable[[np.ndarray], float]] = None
    ) -> None:
        """
        Plot multiple data series and save the plot as a PNG file.

        Args:
            data_list (List[np.ndarray]): List of data arrays to plot.
            name_list (List[str]): Corresponding labels for each data array.
            file_name (str): Filename (without extension) to save the plot.
            add_border (Callable[[np.ndarray], float], optional): Function to determine border line on plot.
        """
        q1 = np.min(np.quantile(data_list, 0.25, axis=1))
        q3 = np.max(np.quantile(data_list, 0.75, axis=1))
        diff = q3 - q1

        for data, label in zip(data_list, name_list):
            plt.plot(data, label=label)
            if add_border:
                plt.axhline(add_border(data), color='black', linestyle='--')

        plt.ylim([max(q1 - 0.5 * diff, 0), q3 + 0.5 * diff] if diff else [q1 - 0.01, q1 + 0.01])
        plt.legend()
        plt.savefig(self.folder_path / f"{file_name}.png")
        plt.clf()

    def save_csv(
        self,
        train: list[list[float]],
        test: list[list[float]],
        name: str
    ) -> None:
        """
        Save training and testing loss lists to a CSV file with columns:
        Epoch, train_loss0, train_loss1, ..., test_loss0, test_loss1

        Args:
            train (list[list[float]]): Training loss values.
            test (list[list[float]]): Testing loss values.
            name (str): Filename (without extension) to save the CSV.
        """
        import pandas as pd

        max_len = max(max(len(t) for t in train), max(len(t) for t in test))
        data = {'Epoch': list(range(max_len))}

        for i, values in enumerate(train):
            padded = values + [None] * (max_len - len(values))
            data[f"train_loss{i}"] = padded

        for i, values in enumerate(test):
            padded = values + [None] * (max_len - len(values))
            data[f"test_loss{i}"] = padded

        df = pd.DataFrame(data)
        df.to_csv(self.folder_path / f"{name}.csv", index=False)

    def save_status(
        self,
        model,
        status,
        extra_info: str = ""
    ) -> None:
        """
        Save model string representation and training status.

        Args:
            model: PyTorch model to convert to string.
            status: Object containing training status attributes.
            extra_info (str): Additional information to save.
        """
        with open(self.folder_path / "session_info.txt", 'w') as f:
            f.write(str(model) + "\n" + extra_info)

        with open(self.folder_path / "status.json", 'w') as f:
            json.dump(vars(status), f, indent=4, sort_keys=True)

    def save_model(
        self,
        model: torch.nn.Module,
        new_val: torch.Tensor,
        epoch: int,
        force_save_last: bool = True
    ) -> bool:
        """
        Save model checkpoint if it improves over previous best.

        Args:
            model (torch.nn.Module): PyTorch model to save.
            new_val (torch.Tensor): New validation metric vector.
            epoch (int): Epoch number for tracking.
            force_save_last (bool): Whether to always save last model regardless of performance.

        Returns:
            bool: True if best model updated and saved, False otherwise.
        """
        new_val = torch.clamp(new_val.float(), 0.0, 1.0)

        if self.best_model_val is None:
            is_better = True
        else:
            weighted_new = self.minimize * new_val
            weighted_best = self.minimize * self.best_model_val * self.b_weights
            is_better = torch.all((weighted_new > weighted_best)[self.b_weights != 0])

        if is_better:
            self.best_model_val = new_val.clone()
            self.best_model_epoch = epoch
            torch.save(model.state_dict(), self.folder_path / "best_model.pt")
            self._save_best_info({"epoch": epoch, "val": new_val.detach().cpu().tolist()})
            return True

        if force_save_last:
            torch.save(model.state_dict(), self.folder_path / "last_model.pt")

        return False

    def _save_best_info(
        self,
        info
    ):
        """
        Save metadata of best model.
        """
        with open(self.folder_path / "best_info.json", 'w') as f:
            json.dump(info, f)

    def exit_handler(
        self
    ):
        """
        Prompt user to save or delete current workspace.
        """
        print("Exiting")
        if input("Do you want to save the current workspace? (y/n) ").lower() == "n":
            shutil.rmtree(self.folder_path)
            print("Deleted workspace")
