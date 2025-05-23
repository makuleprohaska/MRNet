import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

###This part was added to try with the MRNet3 model
class Dataset3(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict, device):
        super().__init__()
        self.device = device
        self.data_dir_axial = f"{data_dir}/axial"
        self.data_dir_coronal = f"{data_dir}/coronal"
        self.data_dir_sagittal = f"{data_dir}/sagittal"

        self.paths_axial = [os.path.join(self.data_dir_axial, file) for file in file_list]
        self.paths_coronal = [os.path.join(self.data_dir_coronal, file) for file in file_list]
        self.paths_sagittal = [os.path.join(self.data_dir_sagittal, file) for file in file_list]
        
        self.paths = [self.paths_axial, self.paths_coronal, self.paths_sagittal]
        
        self.labels = [labels_dict[file] for file in file_list]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy).to(self.device)
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=weights_tensor)
        return loss

    def __getitem__(self, index):
        vol_list = []

        for i in range(3):           
            path = self.paths[i][index]
            vol = np.load(path).astype(np.int32)

            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
            vol = (vol - MEAN) / STDDEV
            vol = np.stack((vol,) * 3, axis=1)

            vol_tensor = torch.FloatTensor(vol).to(self.device)


            vol_list.append(vol_tensor)

        
        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)

        return vol_list, label_tensor

    def __len__(self):
        return len(self.labels)


def load_data3(device, data_dir, labels_csv):
    # Read the CSV without a header, assign column names
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    # Add leading zeros to match the .npy filenames (e.g., 0 -> 0000.npy)
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    # List all .npy files in the data directory
    all_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy")]
    # Filter to only include files that have labels
    all_files = [f for f in all_files if f in labels_dict]
    all_files.sort()

    # Extract labels for stratification
    labels = [labels_dict[file] for file in all_files]

    # Split the data with stratification
    train_files, valid_files = train_test_split(
        all_files, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )

    train_dataset = Dataset3(data_dir, train_files, labels_dict, device)
    valid_dataset = Dataset3(data_dir, valid_files, labels_dict, device)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

    return train_loader, valid_loader