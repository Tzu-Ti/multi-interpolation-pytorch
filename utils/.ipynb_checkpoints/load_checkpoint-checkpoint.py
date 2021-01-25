import torch
import os

def loading(path, network):
    if os.path.isfile(path):
        print('Loading checkpoint...')
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        pretrained_epoch = checkpoint['epoch']
        mask_probability = checkpoint['mask_probability']
        # model loading weight
        network.load_checkpoint(model_state_dict, optimizer_state_dict)
    else:
        raise "Checkpoint path is not valid"
        
    return network