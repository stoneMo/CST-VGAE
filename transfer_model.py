import torch

def transfer_model(pretrained_file, model, device):
    
    pretrained_dict = torch.load(pretrained_file, map_location=device)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # updata model dict
    model.load_state_dict(model_dict)
    return model

def transfer_state_dict(pretrained_dict, model_dict):
    
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict