import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from DIDL_model import DIDL
from Importdata import importdata, generate_interaction_triples  
from torch_geometric.loader import DataLoader as DataLoader_n
from metrics import evaluate
from model import train_one_epoch,predict_one_epoch, compute_bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Training configuration for DIDL model')

# Dataset and File Paths
parser.add_argument('--data_name',         dest='data_name',      type=str,   default='dataset/miRNAmRNA.csv', help='Input dataset file name')
parser.add_argument('--negative_sampling', dest='neg_sampling',   type=float, default=1.0, help='Negative sampling ratio (prevalence)')

# Model Architecture
parser.add_argument('--mir_layer',         dest='mir_layer',      type=list,  default=[64, 32, 20], help='Layer sizes for miRNA encoder')
parser.add_argument('--prot_layer',        dest='prot_layer',     type=list,  default=[64, 32, 20], help='Layer sizes for protein encoder')

parser.add_argument('--dropout',           dest='dropout',        type=float, default=0.5, help='Dropout probability')
parser.add_argument('--reg_L2',            dest='reg_L2',         type=float, default=8e-2, help='L2 regularization coefficient')

# Training Hyperparameters
parser.add_argument('--batch_size',        dest='batch_size',     type=int,   default=32, help='Batch size for training')
parser.add_argument('--n_fold',            dest='n_fold',         type=int,   default=10, help='Number of folds for cross-validation')

# Optimization Parameters
parser.add_argument('--learning_rate',     dest='prt_lr',         type=float, default=1e-5, help='Learning rate for optimizer')
parser.add_argument('--epochs',            dest='epochs',         type=int,   default=20, help='Total epochs for training')
parser.add_argument('--n_epochs_stop',     dest='n_epochs_stop',  type=int,   default=5, help='Number of epochs without improvement to stop training')
parser.add_argument('--epochs_no_improve', dest='epochs_wait',    type=int,   default=0, help='Counter for epochs with no improvement')


args = parser.parse_args()

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '..')

# Initialize dataset loader
dataset_loader = importdata(base_dir, args.data_name)

# Load interactions and entity counts
interactions, [num_miRNAs, num_proteins] = dataset_loader.get_data(base_dir, args.data_name)

# Extract interaction matrix
interaction_matrix = dataset_loader.get_embedding(interactions)

# Generate labeled interaction triples (miRNA, protein, label)
interaction_dataset = generate_interaction_triples(interaction_matrix, args.neg_sampling)


# Lists for tracking loss and AUC during training
train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

# Stores per-fold evaluation metrics for k-fold CV
metrics_cv = {
    "AUC": [],
    "AUPRC": [],
    "Precision": [],
    "Recall": [],
    "Accuracy": [],
    "Specificity": [],
    "NPV": [],
    "F1": []
}

# K-Fold Cross-Validation
kfold = KFold(n_splits=args.n_fold, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(interaction_dataset)):
    print(f"\n===== FOLD {fold} / {args.n_fold} =====")
    print(f"Train samples: {len(train_ids)} | Test samples: {len(test_ids)}")
    print("-"*30)

    # Subset samplers
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # DataLoaders
    train_loader = DataLoader_n(
        interaction_dataset,
        batch_size=args.batch_size,
        sampler=train_subsampler
    )
    test_loader = DataLoader_n(
        interaction_dataset,
        batch_size=args.batch_size,
        sampler=test_subsampler
    )


    # Prepare interaction matrix for training    
    train_list = [interaction_dataset[i] for i in train_ids]
    test_list = [interaction_dataset[i] for i in test_ids]
    train_matrix = dataset_loader.get_embedding(train_list) # interaction matrix for training


    # Initialize model and optimizer
    model = DIDL(torch.tensor(train_matrix, dtype=torch.float32),
                 decoder_type='cosin',
                 mir_enc=args.mir_layer,
                 prot_enc=args.prot_layer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.prt_lr)
    
    # Trackers
    best_auc, min_loss = 0.0, float('inf')
    best_auc_epoch, min_loss_epoch = 0, 0
    wait_counter = 0
    
    # Run the training loop for defined number of epochs
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*30)
        
        # --- Training ---
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        y_train, y_pred_train = predict_one_epoch(model, device, train_loader)
        loss_train = compute_bce_loss(y_train, y_pred_train)
        auc_train = roc_auc_score(y_train, y_pred_train)
        train_losses.append(loss_train)
        train_aucs.append(auc_train)
        print(f"Train Loss: {loss_train:.4f} | Train AUC: {auc_train:.4f}")

        
        # --- Validation ---
        y_val, y_pred_val = predict_one_epoch(model, device, test_loader)
        loss_val = compute_bce_loss(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_pred_val)
        val_losses.append(loss_val)
        val_aucs.append(auc_val)
        print(f"Val Loss:   {loss_val:.4f} | Val AUC:   {auc_val:.4f}")


        # --- Save best model ---
        if auc_val > best_auc:
            best_auc = auc_val
            best_auc_epoch = epoch
            torch.save(model.state_dict(), os.path.join(current_dir, '..', "model", f"best_DIDL_model_fold{fold}.pth"))
        if loss_val < min_loss:
            min_loss = loss_val
            min_loss_epoch = epoch
            wait_counter = 0
        else:
            wait_counter += 1

        # --- Early stopping ---
        if epoch > 5 and wait_counter >= args.n_epochs_stop:
            print("Early stopping triggered.")
            break
        

    #print(f'min_val_loss : {min_loss} for epoch {min_loss_epoch} ............... best_val_auc : {best_auc} for epoch {best_auc_epoch}')
    #print("Model saved")

    
    # --- Evaluate fold ---
    auc, auprc, precision, recall, accuracy, specificity, npv, f1 = evaluate(y_val, y_pred_val)

    # Append metrics
    metrics_cv["AUC"].append(auc)
    metrics_cv["AUPRC"].append(auprc)
    metrics_cv["Precision"].append(precision)
    metrics_cv["Recall"].append(recall)
    metrics_cv["Accuracy"].append(accuracy)
    metrics_cv["Specificity"].append(specificity)
    metrics_cv["NPV"].append(npv)
    metrics_cv["F1"].append(f1)  
    
    
    # --- Plot loss & AUC ---
    epochs_range = range(1, epoch + 1)
    plt.figure(figsize=(7, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'Loss Curve (Fold {fold})')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_aucs, label='Train AUC')
    plt.plot(epochs_range, val_aucs, label='Val AUC')
    plt.ylim([0.5, 1])
    plt.legend()
    plt.title(f'AUC Curve (Fold {fold})')

    plt.tight_layout()
    plt.show()

    # Reset trackers
    train_losses.clear()
    val_losses.clear()
    train_aucs.clear()
    val_aucs.clear()
    

# Print mean and std for each metric
print("\n===== Final 10-Fold Cross-Validation Results =====")
for metric, values in metrics_cv.items():
    print(f"{metric:<12}: {np.mean(values):.4f} ± {np.std(values):.4f}")

