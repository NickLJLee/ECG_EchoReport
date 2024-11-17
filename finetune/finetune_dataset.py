import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import wfdb
from scipy.io import loadmat

'''
In this code:
PTB-XL has four subset: superclass, subclass, form, rhythm
ICBEB is CPSC2018 dataset mentioned in the original paper
Chapman is the CSN dataset from the original paper
'''

class ECGDataset(Dataset):
    def __init__(self, data_path, csv_file, mode='train', dataset_name='I313_I314', backbone='resnet18'):
        """
        Args:
            data_path (string): Path to store raw data.
            csv_file (string): Path to the .csv file with labels and data path.
            mode (string): ptbxl/icbeb/chapman.
        """
        self.dataset_name = dataset_name

        # if self.dataset_name == 'ptbxl':
        #     self.labels_name = list(csv_file.columns[6:])
        #     self.num_classes = len(self.labels_name)

        #     self.data_path = data_path
        #     self.ecg_path = csv_file['filename_hr']
        #     # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
        #     self.labels = csv_file.iloc[:, 6:].values

        if self.dataset_name == 'I313_I314':
            self.labels_name = list(csv_file.columns[11:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 11:].values            

        elif self.dataset_name == 'I_num_2':
            self.labels_name = list(csv_file.columns[11:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 11:].values

        elif self.dataset_name == 'xinbaojiye_ECG':
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 3:].values

        else:
            raise ValueError("dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # if self.dataset_name == 'I313_I314':
            ecg_path = os.path.join(self.data_path, self.ecg_path[idx])
            # the wfdb format file include ecg and other meta data
            # the first element is the ecg data
            data = [wfdb.rdsamp(ecg_path)]
            data = np.array([signal for signal, meta in data])
            data = np.nan_to_num(data, nan=0)
            data = data.squeeze(0) 
            ecg = np.transpose(data,  (1, 0))
            # the raw ecg shape is (5000, 12)
            # transform to (12, 5000)
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
        
            return ecg, target

class ECG_1d_Dataset(Dataset):
    def __init__(self, data_path, csv_file, mode='train', dataset_name='I313_I314', backbone='resnet18'):
        """
        Args:
            data_path (string): Path to store raw data.
            csv_file (string): Path to the .csv file with labels and data path.
            mode (string): ptbxl/icbeb/chapman.
        """
        self.dataset_name = dataset_name
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]
        # if self.dataset_name == 'ptbxl':
        #     self.labels_name = list(csv_file.columns[6:])
        #     self.num_classes = len(self.labels_name)

        #     self.data_path = data_path
        #     self.ecg_path = csv_file['filename_hr']
        #     # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
        #     self.labels = csv_file.iloc[:, 6:].values

        if self.dataset_name == 'I313_I314':
            self.labels_name = list(csv_file.columns[11:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 11:].values            

        elif self.dataset_name == 'I_num_2':
            self.labels_name = list(csv_file.columns[11:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 11:].values

        elif self.dataset_name == 'xinbaojiye_ECG':
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)

            self.data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
            self.ecg_path = csv_file['waveform_path']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 3:].values

        else:
            raise ValueError("dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # if self.dataset_name == 'I313_I314':
            ecg_path = os.path.join(self.data_path, self.ecg_path[idx])
            # the wfdb format file include ecg and other meta data
            # the first element is the ecg data
            data = [wfdb.rdsamp(ecg_path)]
            data = np.array([signal for signal, meta in data])
            data = np.nan_to_num(data, nan=0)
            data = data.squeeze(0) 
            data = np.transpose(data,  (1, 0))
            # the raw ecg shape is (5000, 12)
            # transform to (12, 5000)
            # normalzie to 0-1
            ecg = data[self.lead_indices, :]
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
        
            return ecg, target


def getdataset(data_path, csv_path, mode='train', dataset_name='I313_I314', ratio=100, backbone='resnet18'):
    ratio = int(ratio)

    if dataset_name == 'I313_I314':
        csv = pd.read_csv(csv_path)
    elif dataset_name == 'I_num_2':
        csv = pd.read_csv(csv_path)
    elif dataset_name == 'xinbaojiye_ECG':
        csv = pd.read_csv(csv_path)        
    
    csv.reset_index(drop=True, inplace=True)

    dataset = ECGDataset(data_path, csv, mode=mode, dataset_name=dataset_name,backbone=backbone)

    return dataset

def get_1d_dataset(data_path, csv_path, mode='train', dataset_name='I313_I314', ratio=100, backbone='resnet18'):
    ratio = int(ratio)

    if dataset_name == 'I313_I314':
        csv = pd.read_csv(csv_path)
    elif dataset_name == 'I_num_2':
        csv = pd.read_csv(csv_path)
    elif dataset_name == 'xinbaojiye_ECG':
        csv = pd.read_csv(csv_path)  

    csv.reset_index(drop=True, inplace=True)

    dataset = ECG_1d_Dataset(data_path, csv, mode=mode, dataset_name=dataset_name,backbone=backbone)

    return dataset
