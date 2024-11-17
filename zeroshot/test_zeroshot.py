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

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda'

config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

model = utils_builder.ECGCLIP(config['network'])
ckpt = '/data1/1shared/lijun/ecg/ECG-EchoReport/checkpoints/1_lead_model_bestZeroShotAll_ckpt.pth'
ckpt = torch.load(f'{ckpt}', map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device_id)
model = torch.nn.DataParallel(model)

args_zeroshot_eval = config['zeroshot']

avg_sens, avg_spec, avg_auc = 0, 0, 0
for set_name in args_zeroshot_eval['test_sets'].keys():

        sens, spec, auc, _, _, _, res_dict = \
        zeroshot_eval(model=model, 
        set_name=set_name, 
        device=device_id, 
        args_zeroshot_eval=args_zeroshot_eval)

        avg_sens += sens
        avg_spec += spec
        avg_auc += auc

avg_sens = avg_sens/len(args_zeroshot_eval['test_sets'].keys())
avg_spec = avg_spec/len(args_zeroshot_eval['test_sets'].keys())
avg_auc = avg_auc/len(args_zeroshot_eval['test_sets'].keys())