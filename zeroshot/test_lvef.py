import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error  # for MAE computation
import yaml as yaml
import sys
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.model_selection import train_test_split

import os
import random
import yaml as yaml
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval

class LVEF_reg_Dataset(Dataset):
    def __init__(self, labels_df, transform=None):
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 5])
        labels = self.labels_df.iloc[idx, 3]  # LVEF ground truth label
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels

def compute_regression_metric(
    ecg_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    per_frame_similarities = (
        ecg_embeddings @ prompt_embeddings
    )

    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    prompt_values = torch.tensor(
        prompt_values, device=ecg_embeddings.device
    )

    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(dim=0)
    # print(prompt_embeddings.shape)
    # print("all:",all_frames_ranked_values)
    # print("avg:",avg_frame_ranked_values)
    twenty_percent = int(avg_frame_ranked_values.shape[0] * 0.8)
    top_twenty_percent_values = avg_frame_ranked_values[:twenty_percent]
    final_prediction = top_twenty_percent_values.median()

    return final_prediction

def lvef_reg(model, loader, device='cuda'):
    zero_shot_prompts = {
        "ejection_fraction": [
            "LVEF = <#>% ",
            # "LVEF > <#>% ",
        ],
    }

    ejection_fraction_prompts = zero_shot_prompts["ejection_fraction"]

    prompts = []
    prompt_values = []
    prompt_embeddings = []
    model.eval()
    for prompt in ejection_fraction_prompts:
        for i in range(101):
            prompts.append(prompt.replace("<#>", str(i)))
            prompt_values.append(i)

    for prompt in tqdm(prompts):
        prompt = [prompt]
        ejection_fraction_prompts = model.module._tokenize(prompt)
        class_embeddings = model.module.get_text_emb(
            ejection_fraction_prompts.input_ids.to(device=device),
            ejection_fraction_prompts.attention_mask.to(device=device)
        )
        class_embeddings = model.module.proj_t(class_embeddings)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        prompt_embeddings.append(class_embedding)

    ecg_embeddings = []
    ejection_fraction_predictions = []
    ground_truths = []  # Store ground truth values

    with torch.no_grad():
        for i, (ecg, target) in enumerate(tqdm(loader)):
            ecg = ecg.to(device=device)
            ecg_emb = model.module.ext_ecg_emb(ecg)
            ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)
            ecg_embeddings.append(ecg_emb)
            ground_truths.extend(target.numpy())  # Collect ground truth labels

    ecg_embeddings = torch.cat(ecg_embeddings, dim=0)
    prompt_embeddings = torch.stack(prompt_embeddings, dim=1)

    for i in range(ecg_embeddings.shape[0]):
        ejection_fraction_prediction = compute_regression_metric(
            ecg_embeddings[i,:].unsqueeze(0), prompt_embeddings, torch.tensor(prompt_values).to(device=device)
        )
        ejection_fraction_predictions.append(ejection_fraction_prediction)

    return ejection_fraction_predictions, ground_truths


# Main execution block
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device_id = 'cuda:0'

config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)

model = utils_builder.ECGCLIP(config['network'])
ckpt = '/data1/1shared/lijun/ecg/ECG-EchoReport/checkpoints/I313_I314_zero-shot_10_ckpt.pth'
ckpt = torch.load(f'{ckpt}', map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device_id)
model = torch.nn.DataParallel(model)

df_label = '/home/lijun/code/LVEF.csv'
df_label = pd.read_csv(df_label)
train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

test_dataset = LVEF_reg_Dataset(labels_df=test_df)
testloader = DataLoader(test_dataset, batch_size=256, num_workers=40, shuffle=False)

ejection_fraction_predictions, ground_truths = lvef_reg(model=model, loader=testloader)

# Compute Mean Absolute Error (MAE)
ground_truths = np.array(ground_truths)
# Convert predictions to numpy arrays and ensure they are 1D arrays before concatenation
# Ensure each prediction is wrapped into a 1D array before appending to the list
ejection_fraction_predictions = [np.expand_dims(pred.cpu().numpy(), axis=0) if isinstance(pred, torch.Tensor) else np.expand_dims(np.array(pred), axis=0) for pred in ejection_fraction_predictions]

# Now concatenate the list of NumPy arrays into a single array
ejection_fraction_predictions = np.concatenate(ejection_fraction_predictions, axis=0)

# Convert ground_truths to NumPy if it's not already
ground_truths = np.array(ground_truths)

# Create a DataFrame for saving to CSV
df = pd.DataFrame({
    'Ground_Truth_LVEF': ground_truths.flatten(),  # Flatten to ensure it's 1D
    'Predicted_LVEF': ejection_fraction_predictions.flatten()  # Flatten if needed
})
output_file = '/data1/1shared/lijun/ecg/ECG-EchoReport/res/LVEF_output.csv'
df.to_csv(output_file, index=False)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(ground_truths, ejection_fraction_predictions)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


