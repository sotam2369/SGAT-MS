import torch.nn as nn
import torch
import torch.optim as optim
from torch_geometric.utils import scatter
from loss import SGATLoss

class SGATTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_cls,
        device: torch.device,
        train_loader,
        val_loader,
        loss_fn: SGATLoss,
        cut_incomplete_batch: bool = False,
        learning_rate: float = None,
        randomize_input_prob: float = 0.1,
    ):
        """
        Initialize the SGATTrainer with model, optimizer, data loaders, loss function, and training parameters.
        """
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.cut_incomplete_batch = cut_incomplete_batch
        self.learning_rate = learning_rate
        self.random_input = RandomInput(randomize_input_prob)

    def pre_training(self):
        """
        Prepare the model and optimizer before training starts.
        Moves model to device, sets up optimizer and scheduler if learning rate is specified.
        """
        self.model.to(self.device)

        if self.learning_rate is not None:
            self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = self.optimizer_cls(self.model.parameters())

    def train_epoch(
        self,
        fixed_input=None,
        compute_cost: bool = False,
    ):
        """
        Perform a single training epoch over the training dataset.

        Args:
            fixed_input: Optional fixed input features to replace data.x.
            return_output: If True, do not backpropagate and return model outputs.
            compute_cost: If True, compute and return cost metrics per instance.

        Returns:
            Tuple of average loss tensor, average evaluation tensor,
            and optionally output list and cost tensor if compute_cost is True.
        """
        self.model.train()

        total_loss = None
        total_eval = None
        batch_size = self.train_loader.batch_size or 1

        if compute_cost:
            out_list = []
            cost_list = []

        total_num = 0

        for data in self.train_loader:
            if self.cut_incomplete_batch and data.num_graphs != batch_size:
                break

            total_num += data.num_graphs

            self.optimizer.zero_grad()

            data = self.random_input(data)
            data = data.to(self.device)

            if fixed_input is not None:
                data.x = fixed_input

            output = self.model(data)

            loss = self.loss_fn(output, data)
            loss_mean = loss.mean()
            if total_loss is None:
                total_loss = torch.zeros_like(loss_mean)
            total_loss += loss_mean * data.num_graphs

            eval_out = self.loss_fn.evaluation(output, data)
            if total_eval is None:
                total_eval = torch.zeros_like(eval_out)
            total_eval += eval_out * data.num_graphs

            loss_mean.backward()
            self.optimizer.step()

            if compute_cost:
                return_out = output[1].detach()

                for i in range(batch_size):
                    out_list.append(return_out[data.batch[data.mask == 0] == i])

                clause_mask = data.mask == 1
                clause_batch = data.batch[clause_mask]
                unsatisfied = torch.round(1 - output[2][clause_mask, 0].detach())
                cost_int = scatter(
                    unsatisfied * data.weights,
                    clause_batch,
                    dim=0,
                    dim_size=data.num_graphs,
                    reduce='sum'
                )

                cost_list.append(cost_int.detach())

        if total_loss is None:
            total_loss = torch.zeros((), device=self.device)
        if total_eval is None:
            total_eval = torch.zeros((), device=self.device)

        average_loss = total_loss / total_num
        average_eval = total_eval / total_num

        if compute_cost:
            return average_loss, average_eval, out_list, torch.cat(cost_list)

        return average_loss, average_eval

    def _prepare_data(self, data, fixed_input):
        """
        Helper method to prepare data batch by moving to device and applying random input perturbation.

        Args:
            data: Batch data object.
            fixed_input: Optional fixed input features to replace data.x.

        Returns:
            Processed data batch on the correct device.
        """
        data = data.to(self.device)
        data = self.random_input(data)

        if fixed_input is not None:
            data.x = fixed_input

        return data

    def val_epoch(
        self,
        fixed_input=None,
        include_train: bool = False,
    ):
        """
        Perform a validation epoch over the validation dataset, optionally including training data evaluation.

        Args:
            fixed_input: Optional fixed input features to replace data.x.
            include_train: If True, also evaluate on training data.

        Returns:
            Tuple of average loss tensor, average evaluation tensor, and list of instance names.
        """
        self.model.eval()

        with torch.no_grad():
            total_loss = None
            total_eval = None

            for data in self.val_loader:
                data = self._prepare_data(data, fixed_input)

                output = self.model(data)
                loss = self.loss_fn(output, data)
                eval_out = self.loss_fn.evaluation(output, data)

                if total_eval is None:
                    total_eval = torch.zeros_like(eval_out)
                total_eval += eval_out * data.num_graphs
                loss_mean = loss.mean()
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_mean)
                total_loss += loss_mean * data.num_graphs

            if include_train:
                for data in self.train_loader:
                    data = self._prepare_data(data, fixed_input)

                    output = self.model(data)
                    eval_out = self.loss_fn.evaluation(output, data)

                    total_eval += eval_out

        total_instances = len(self.val_loader.dataset) + (len(self.train_loader.dataset) if include_train else 0)

        if total_loss is None:
            total_loss = torch.zeros((), device=self.device)
        if total_eval is None:
            total_eval = torch.zeros((), device=self.device)

        return total_loss / total_instances, total_eval / total_instances

    def inference(self, instance, detach: bool = False):
        """
        Perform inference on a single instance.

        Args:
            instance: Single data instance to run inference on.
            detach: If True, detach outputs from computation graph and move to CPU.

        Returns:
            Model output tensor(s).
        """
        self.model.eval()

        instance = instance.to(self.device)
        output = self.model(instance)

        if detach:
            output = [x.detach().cpu() for x in output]

        return output

    def get_cost(self, instance, use_weights: bool = False, detach: bool = False):
        """
        Compute the cost of a given instance based on model output and constraints.

        Args:
            instance: Data instance to compute cost for.
            use_weights: If True, use instance weights in cost calculation.
            detach: If True, detach outputs and cost from computation graph and move to CPU.

        Returns:
            Tuple of (cost, model output)
            - cost: Computed cost value.
            - model output: Output tensor(s) from model.
        """
        with torch.no_grad():
            self.model.eval()

            instance = instance.to(self.device)
            output = self.model(instance)
            output_rounded = torch.round(1 - output[2][instance.mask == 1, 0])

            if use_weights:
                cost = (output_rounded * instance.weights).sum()
            else:
                cost = output_rounded.sum()

            if detach:
                output = [x.detach().cpu() for x in output]
                cost = cost.detach().cpu()

        return cost, output


class RandomInput:
    def __init__(self, p=0.1):
        """
        Initialize RandomInput with probability p of randomizing input features per graph.
        """
        self.p = p

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Randomly replace input features in a sample with probability p per graph.

        Args:
            sample: Batch data object.

        Returns:
            Modified sample with randomized inputs in some graphs.
        """
        for batch in range(sample.num_graphs):
            if torch.rand(1) < self.p:
                sample.x[sample.batch == batch] = torch.rand_like(sample.x[sample.batch == batch])

        return sample
