import torch.nn.init as init

import os
import numpy as np
  
def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def initialize_model(model):
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    return model

def filter_parameter(model):
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    
    return filtered_parameters

def encode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province[idx]
        if prov in text:
            text = text.replace(prov, province_replace[idx])

    return text

def decode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province_replace[idx]
        if prov in text:
            text = text.replace(prov, province[idx])
    return text