import argparse
import random
import atexit
import json
import os

import torch
import numpy as np
from tqdm import tqdm

from manager.folder_manager import SGATFolder
from manager.dataset_manager import SGATDataset
from models import SGAT
from trainer import SGATTrainer, SGATLoss


def getArgs():
    parser = argparse.ArgumentParser(description='SGAT: A Satisfiability based Graph Attention Network')

    parser.add_argument('-dtrain', '--dataset-train', default='../dataset/processed/train_2000.pkl', type=str, help='Path to training dataset')
    parser.add_argument('-dtest',  '--dataset-test',  default='../dataset/processed/test_200.pkl',  type=str, help='Path to testing dataset')
    parser.add_argument('-trs',    '--train-split',  default=0,                           type=int, help='Use split datasets for training')
    parser.add_argument('-tes',    '--test-split',   default=0,                           type=int, help='Use split datasets for testing')
    parser.add_argument('-dir',    '--dir',          default='../plots/',                type=str, help='Directory to save the model')

    parser.add_argument('-ly',     '--layers',       default=2,                           type=int, help='Number of layers to use')
    parser.add_argument('--heads',                 default=8,                           type=int, help='Number of heads to use')
    parser.add_argument('--hidden',                default=1,                           type=int, help='Hidden size to use')
    parser.add_argument('-tn',     '--t-norm',       default=None,                        type=str, choices=["godel", "product", "lukasiewicz", "soft_godel", "einstein"], help='Type of T-Norm to use')
    parser.add_argument('-ed',     '--edge_dimension',default=4,                           type=int, help='Dimension of edge features')

    parser.add_argument('-c',      '--cuda',         default=None,                        type=str, help='Cuda device to use, if available')
    parser.add_argument('-e',      '--epochs',       default=100,                         type=int, help='Number of epochs')
    parser.add_argument('-b',      '--batch-size',   default=64,                          type=int, help='Batch size for training')
    parser.add_argument('--test-batch-size',         default=1,                           type=int, help='Batch size for testing')
    parser.add_argument('-s',      '--seed',         default=1,                           type=int, help='Random seed')

    parser.add_argument('-oe',     '--output-epochs',default=10,                          type=int, help='Number of epochs to output')
    parser.add_argument('-fr',     '--finish-round', default=1000,                        type=int, help='Number of rounds to finish')

    parser.add_argument('-lr',     '--lr',           default=0.0001,                      type=float, help='Learning rate')
    parser.add_argument('-cl',     '--cut-last',     action='store_true',                 help='Cut last batch or not')
    parser.add_argument('-opt',    '--optimizer',    default='NAdam',                     type=str, help='Optimizer to use')

    parser.add_argument('--id',                default=None,                          type=int, help='ID of the model in case of resuming training')
    parser.add_argument('--best-weights',      default=[1.0],                        type=float, nargs=1, help='Weights for evaluation')
    parser.add_argument('--trans-prob',        default=0.1,                          type=float, help='Transformation probability')
    parser.add_argument('--normalization',     action='store_true',                   help='No normalization')
    parser.add_argument('--dropout',           default=0,                            type=float, help='Dropout rate')
    parser.add_argument('--use-gat',           action='store_true',                   help='Use standard GATConv instead of SGATv2Conv')
    parser.add_argument('--loss-type', default=None, type=str, choices=['bce', 'mse'], help='Loss function to use: "bce" or "mse"')

    return parser.parse_args()


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _fmt(value):
    return f"{value:.6g}"


def main(args):

    # Set device
    device = torch.device(args.cuda if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    best_weights = args.best_weights
    if isinstance(best_weights, torch.Tensor):
        best_weights = best_weights.detach().cpu().tolist()
    if not isinstance(best_weights, list):
        best_weights = [best_weights]
    if len(best_weights) == 0:
        best_weights = [1.0]
    else:
        best_weights = [best_weights[0]]
    args.best_weights = best_weights

    # Initialize working folder
    if args.id is not None:
        folder_path = os.path.join(args.dir, f'train_{args.id}/')
        work_folder = SGATFolder(folder_path=folder_path, b_weights=args.best_weights)
        # Load status.json manually
        with open(os.path.join(folder_path, 'status.json'), 'r') as f:
            ns = json.load(f)
        args = argparse.Namespace(**ns)
    else:
        work_folder = SGATFolder(b_weights=args.best_weights, directory=args.dir)

    atexit.register(work_folder.exit_handler)

    # Load dataset
    dataset = SGATDataset()
    dataset.load_from_pickle(
        train_path=args.dataset_train,
        test_path=args.dataset_test,
        train_split=args.train_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )
    print(f"Dataset Loaded. Train: {len(dataset.train.dataset)} Test: {len(dataset.test.dataset)}")
    setSeed(args.seed)

    # Setup loss function
    if args.loss_type == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss
    elif args.loss_type == 'mse':
        loss_fn = torch.nn.MSELoss
    else:
        loss_fn = torch.nn.MSELoss
    loss = SGATLoss(device=device, loss_type=loss_fn)

    # Setup model
    model = SGAT(
        iterations=args.layers,
        heads=args.heads,
        t_norm=args.t_norm,
        normalization=args.normalization,
        d_edge=args.edge_dimension,
        hidden=args.hidden,
        dropout=args.dropout,
        use_gat=args.use_gat
    )
    # Save initial status
    work_folder.save_status(model, args)

    # Setup optimizer
    optimizer = torch.optim.NAdam
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD

    # Setup trainer and pre-training
    trainer = SGATTrainer(model, optimizer, device, dataset.train, dataset.test, loss, cut_incomplete_batch=args.cut_last, randomize_input_prob=args.trans_prob, learning_rate=args.lr)
    trainer.pre_training()

    train_loss_history = []
    test_loss_history = []
    train_eval_history = []
    test_eval_history = []

    # Training loop
    pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", ncols=120)
    for epoch in pbar:

        train_loss, train_eval = trainer.train_epoch()

        loops = 1
        test_results = [trainer.val_epoch() for _ in range(loops)]
        test_loss = sum(res[0] for res in test_results) / loops
        test_eval = sum(res[1] for res in test_results) / loops

        train_loss_value = float(train_loss.detach().cpu())
        test_loss_value = float(test_loss.detach().cpu())
        train_eval_value = float(train_eval.detach().cpu())
        test_eval_value = float(test_eval.detach().cpu())

        train_loss_history.append(train_loss_value)
        test_loss_history.append(test_loss_value)
        train_eval_history.append(train_eval_value)
        test_eval_history.append(test_eval_value)

        pbar.set_postfix(
            tr_loss=_fmt(train_loss_value),
            te_loss=_fmt(test_loss_value),
            tr_eval=_fmt(train_eval_value),
            te_eval=_fmt(test_eval_value),
        )
        
        if epoch >= 10:
            if work_folder.save_model(model, test_eval.detach().cpu(), epoch):
                pass

            if epoch % args.output_epochs == 0:
                pbar.write(f"Epoch {epoch}")
                pbar.write(f"  Tr Loss: {_fmt(train_loss_value)}")
                pbar.write(f"  Tr Eval: {_fmt(train_eval_value)}")
                pbar.write(f"  Te Loss: {_fmt(test_loss_value)}")
                pbar.write(f"  Te Eval: {_fmt(test_eval_value)}")
                pbar.write(f"  Best: {work_folder.best_model_val.tolist()}, Epoch: {work_folder.best_model_epoch}")
                window = min(len(test_eval_history), 10)
                mean_eval = sum(test_eval_history[-window:]) / window if window else 0.0
                pbar.write(f"  Mean Eval per 10 epochs: {_fmt(mean_eval)}")
                
                if len(test_eval_history) > 20:
                    recent = sum(test_eval_history[-10:])
                    previous = sum(test_eval_history[-20:-10])
                    pbar.write(
                        f"  Difference in Mean Eval per 10 epochs: {_fmt((recent - previous) / 10)}"
                    )

                
                work_folder.save_plot(
                    [np.array(train_loss_history), np.array(test_loss_history)],
                    ['Train', 'Test'],
                    'loss'
                )
                work_folder.save_plot(
                    [np.array(train_eval_history), np.array(test_eval_history)],
                    ['Train', 'Test'],
                    "eval",
                    add_border=max
                )
                work_folder.save_csv([train_loss_history], [test_loss_history], "loss")
                work_folder.save_csv([train_eval_history], [test_eval_history], "eval")

                if epoch - work_folder.best_model_epoch > args.finish_round and args.finish_round != -1:
                    pbar.write("Finishing due to no improvement")
                    break
    
    atexit.unregister(work_folder.exit_handler)
    print("Exiting:", work_folder.folder_id)
    exit(work_folder.folder_id)


if __name__ == '__main__':
    args = getArgs()
    main(args)
