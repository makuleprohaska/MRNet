import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model
from loader import load_data3
from model import MRNet3

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train3(rundir, epochs, learning_rate, use_gpu, use_mps, data_dir, labels_csv, weight_decay, max_patience, batch_size, label_smoothing):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    train_loader, valid_loader = load_data3(device, data_dir, labels_csv, batch_size=batch_size, label_smoothing=label_smoothing)
    
    model = MRNet3()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max_patience, factor=.3, threshold=1e-4)

    best_val_auc = float('-inf')

    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader, train=False)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

            file_name = f'val{val_auc:0.4f}_train{train_auc:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            
            print(f"Saving model to {save_path}")
            
            torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    parser.add_argument('--learning_rate', default=1e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training and validation')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing factor')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False 
    elif args.mps and torch.backends.mps.is_available():
        pass

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train3(args.rundir, args.epochs, args.learning_rate, 
           args.gpu, args.mps, args.data_dir, args.labels_csv, args.weight_decay, args.max_patience, args.batch_size, args.label_smoothing)