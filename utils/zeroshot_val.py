import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
import yaml as yaml
import sys
sys.path.append("../finetune/")
from typing import Tuple
from finetune_dataset import getdataset as get_zero_dataset
from finetune_dataset import get_1d_dataset as get_1d_zero_dataset


ecg_pair_template = ("{}", "no {}")

def compute_auc_with_ci(gt_np, pred_np, num_bootstrap=10, alpha=0.05):
    """Calculate AUC and its confidence interval using bootstrapping."""
    fpr, tpr, _ = roc_curve(gt_np, pred_np)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrapping
    rng = np.random.default_rng()
    aucs = []
    for _ in range(num_bootstrap):
        indices = rng.choice(len(gt_np), len(gt_np), replace=True)
        if len(np.unique(gt_np[indices])) < 2:  # Ensure at least one positive and one negative
            continue
        fpr_boot, tpr_boot, _ = roc_curve(gt_np[indices], pred_np[indices])
        aucs.append(auc(fpr_boot, tpr_boot))
    
    # Calculate confidence intervals
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return roc_auc, (lower, upper)


def zeroshot_classifier(classnames, templates, model, context_length=77):
    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] # format with class
    #         texts = clip.tokenize(texts, context_length=context_length) # tokenize
    #         class_embeddings = model.encode_text(texts) # embed with text encoder
            
    #         # normalize class_embeddings
    #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         # average over templates 
    #         class_embedding = class_embeddings.mean(dim=0) 
    #         # norm over new averaged templates
    #         class_embedding /= class_embedding.norm() 
    #         zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    # return zeroshot_weights

def get_class_emd(model, class_name, templates, device='cuda'):
    model.eval()
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in tqdm(class_name):
            # texts = texts.lower()
            texts = [template.format(texts) for template in templates] # format with class
            print(texts)
            # texts = [texts] # convert to list
            texts = model._tokenize(texts) # tokenize
            class_embeddings = model.get_text_emb(texts.input_ids.to(device=device)
                                                            , texts.attention_mask.to(device=device)
                                                            ) # embed with text encoder
            class_embeddings = model.proj_t(class_embeddings) # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def get_ecg_emd(model, loader, zeroshot_weights, device='cuda', softmax_eval=True):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, (ecg, target) in enumerate(tqdm(loader)):
            ecg = ecg.to(device=device) 
            # predict
            ecg_emb = model.ext_ecg_emb(ecg)
            ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)

            # obtain logits (cos similarity)
            logits = ecg_emb @ zeroshot_weights
            logits = torch.squeeze(logits, 0) # (N, num_classes)
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = torch.sigmoid(norm_logits) 
            
            y_pred.append(logits.cpu().data.numpy())
        
    y_pred = np.concatenate(y_pred, axis=0)
    return np.array(y_pred)

def run_single_prediction(target_class, template, model, loader, softmax_eval=True, context_length=77): 

    ecg_phrase = [template]
    zeroshot_weights = get_class_emd(model.module, target_class, ecg_phrase)
    y_pred = get_ecg_emd(model.module, loader, zeroshot_weights, softmax_eval=True)
    return y_pred

def run_softmax_eval(model, loader, target_class, pair_template, context_length= 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos_prompt = pair_template[0]
    neg_prompt = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(target_class, pos_prompt, model, loader, 
                                     softmax_eval=True, context_length=context_length) 
    neg_pred = run_single_prediction(target_class, neg_prompt, model, loader, 
                                     softmax_eval=True, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred

def compute_regression_metric(
    ecg_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    per_frame_similarities = (
        ecg_embeddings @ prompt_embeddings.T
    )  # (N x Frames x Candidates)

    # Sort the candidates by their similarity to the video
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    # Convert matrix of indices to their corresponding continuous values.
    prompt_values = torch.tensor(
        prompt_values, device=ecg_embeddings.device
    )  # (N x Frames x Candidates)
    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]

    # Taking the mean along dim=1 collapses the frames dimension
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(
        dim=1
    )  # (N x Candidates)

    # The median of only the top 20% of predicted values is taken
    # as the final predicted value
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]

    return final_prediction

def zeroshot_eval(model, set_name, device='cuda', args_zeroshot_eval=None):
    assert args_zeroshot_eval is not None, "Please specify the test set!"

    set_name = set_name
    num_workers = args_zeroshot_eval['num_workers']
    batch_size = args_zeroshot_eval['batch_size']

    # meta_data_path = args_zeroshot_eval['meta_data_path']

    # if 'val_sets' not in args_zeroshot_eval.keys():
    #     data_path = args_zeroshot_eval['test_sets'][set_name]['data_path']
    # if 'val_sets' in args_zeroshot_eval.keys():
    #     data_path = args_zeroshot_eval['val_sets'][set_name]['data_path']
    
    data_path = '/hot_data/lijun/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

    # meta_split_path = args_zeroshot_eval['meta_split_path']
    # if 'val_sets' not in args_zeroshot_eval.keys():
    #     split_path = args_zeroshot_eval['test_sets'][set_name]['split_path']
    # if 'val_sets' in args_zeroshot_eval.keys():
    #     split_path = args_zeroshot_eval['val_sets'][set_name]['split_path']
    # split_path = os.path.join(meta_split_path, split_path)


    if 'I313_I314' in set_name:
        split_path = '/data1/1shared/lijun/ecg/ECG-EchoReport/finetune/data_split/I313_I314.csv'
        test_dataset = get_1d_zero_dataset(data_path, split_path, mode='test', dataset_name='I313_I314')
    if 'I_num_2' in set_name:
        split_path = '/data1/1shared/lijun/ecg/ECG-EchoReport/finetune/data_split/I_num_2.csv'
        test_dataset = get_1d_zero_dataset(data_path, split_path, mode='test', dataset_name='I_num_2')
    if 'xinbaojiye_ECG' in set_name:
        split_path = '/data1/1shared/lijun/ecg/ECG-EchoReport/finetune/data_split/xinbaojiye_ECG.csv'
        test_dataset = get_1d_zero_dataset(data_path, split_path, mode='test', dataset_name='xinbaojiye_ECG')

    # else:
    #     test_dataset = get_zero_dataset(data_path, split_path, mode='test', dataset_name=set_name)
    class_name = test_dataset.labels_name

    # open json as dict
    with open(args_zeroshot_eval['prompt_dict'], 'r') as f:
        prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    # get prompt for each class
    target_class = [prompt_dict[i] for i in class_name]
    # print(class_name)
    print('***********************************')
    print('zeroshot classification set is {}'.format(set_name))
    
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

    # get the target array from testset
    gt = test_dataset.labels

    # get class embedding
    zeroshot_weights = get_class_emd(model.module, target_class, ecg_pair_template, device=device)
    # get ecg prediction

    pred = get_ecg_emd(model.module, test_dataloader, zeroshot_weights, device=device, softmax_eval=True)
    # pred = run_softmax_eval(model, test_dataloader, target_class, ecg_pair_template)
    AUROCs = []
    sensitivities = []
    specificities = []
    youden_indices = []

    for i in range(len(target_class)):   
        gt_np = gt[:, i]
        pred_np = pred[:, i]
        
        # Calculate precision, recall, and thresholds for sensitivity (recall) and specificity
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        
        # Calculate the Youden Index for each threshold
        youden_index_values = []
        for thresh in thresholds:
            pred_binary = pred_np > thresh
            tn, fp, fn, tp = confusion_matrix(gt_np, pred_binary).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            youden_index = sensitivity + specificity - 1
            youden_index_values.append(youden_index)
        
        max_youden_index = max(youden_index_values)
        optimal_threshold = thresholds[youden_index_values.index(max_youden_index)]
        
        tn, fp, fn, tp = confusion_matrix(gt_np, pred_np > optimal_threshold).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        sensitivities.append(sensitivity * 100)
        specificities.append(specificity * 100)
        youden_indices.append(max_youden_index * 100)

        # Calculate ROC AUC with confidence interval
        roc_auc, (ci_lower, ci_upper) = compute_auc_with_ci(gt_np, pred_np)
        AUROCs.append(roc_auc * 100)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(gt_np, pred_np)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}])')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {class_name[i]}')
        plt.legend(loc="lower right")
        plt.savefig(f'/data1/1shared/lijun/ecg/ECG-EchoReport/res/AUC_fig/{class_name[i]}.png')

    sensitivity_avg = np.array(sensitivities).mean()    
    specificity_avg = np.array(specificities).mean()
    youden_index_avg = np.array(youden_indices).mean()
    AUROC_avg = np.array(AUROCs).mean()

    res_dict = {
        'AUROC_avg': AUROC_avg,
        'Sensitivity_avg': sensitivity_avg,
        'Specificity_avg': specificity_avg,
        'Youden_Index_avg': youden_index_avg
    }
    for i in range(len(target_class)):
        res_dict.update({
            f'AUROC_{class_name[i]}': AUROCs[i],
            f'Sensitivity_{class_name[i]}': sensitivities[i],
            f'Specificity_{class_name[i]}': specificities[i],
            f'Youden_Index_{class_name[i]}': youden_indices[i]
        })

    print('-----------------------------------')
    print(f'The average AUROC is {AUROC_avg:.4f}')
    for i in range(len(target_class)):
        print(f'The AUROC of {class_name[i]} is {AUROCs[i]:.2f}')
            
    print('-----------------------------------')
    print(f'The average Sensitivity is {sensitivity_avg:.4f}')
    for i in range(len(target_class)):
        print(f'The Sensitivity of {class_name[i]} is {sensitivities[i]:.2f}')

    print('-----------------------------------')
    print(f'The average Specificity is {specificity_avg:.4f}')
    for i in range(len(target_class)):
        print(f'The Specificity of {class_name[i]} is {specificities[i]:.2f}')
        
    print('-----------------------------------')
    # print(f'The average Youden Index is {youden_index_avg:.4f}')
    # for i in range(len(target_class)):
    #     print(f'The Youden Index of {class_name[i]} is {youden_indices[i]:.2f}')
    # print('***********************************')

    return sensitivity_avg, specificity_avg, AUROC_avg, sensitivities, specificities, AUROCs, res_dict


def lvef_reg(model, loader, device='cuda'):
  zero_shot_prompts = {
      "ejection_fraction": [
          "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
          "LV EJECTION FRACTION IS <#>%. ",
      ],
  }

  ejection_fraction_prompts = zero_shot_prompts["ejection_fraction"]
  print(ejection_fraction_prompts)

  # However, since ejection fraction can range between 0 and 100,
  # we'll need to make 100 versions of each prompt.
  prompts = []
  prompt_values = []

  for prompt in ejection_fraction_prompts:
      for i in range(101):
          prompts.append(prompt.replace("<#>", str(i)))
          prompt_values.append(i)

  ejection_fraction_prompts = prompts
  # We'll once again tokenize and embed the prompts
  ejection_fraction_prompts = model._tokenize(ejection_fraction_prompts)
  class_embeddings = model.get_text_emb(ejection_fraction_prompts.input_ids.to(device=device)
                                                  , ejection_fraction_prompts.attention_mask.to(device=device)
                                                  ) # embed with text encoder
  class_embeddings = model.proj_t(class_embeddings) # embed with text encoder
  # normalize class_embeddings
  class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
  # average over templates 
  class_embedding = class_embeddings.mean(dim=0) 
  # norm over new averaged templates
  class_embedding /= class_embedding.norm() 


  model.eval()
  with torch.no_grad():
      for i, (ecg, target) in enumerate(tqdm(loader)):
        ecg = ecg.to(device=device) 
        # predict
        ecg_embedding = model.ext_ecg_emb(ecg)
        ecg_embedding /= ecg_embedding.norm(dim=-1, keepdim=True)

  # And we'll compute the similarity between the image and the prompts
  # to get a prediction for the ejection fraction.
  ejection_fraction_predictions = compute_regression_metric(
      ecg_embedding, class_embedding, prompt_values
  )
  print(f"Predicted ejection fraction is {ejection_fraction_predictions.item():.1f}%")
  
  return ejection_fraction_predictions