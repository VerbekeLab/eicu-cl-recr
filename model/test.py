import os, glob
import wandb

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.model import GRUModel
from model.utils import store_test_metrics
from model.utils import MSLELoss, Metrics, eICU_Loader

def test(args, central=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if central:
        model_dir = max(glob.glob(os.path.join(args.central_model_dir, '*/')), key=os.path.getmtime)
    
    else:
        model_dir = max(glob.glob(os.path.join(args.dist_model_dir, '*/')), key=os.path.getmtime)

    model_path = os.path.join(model_dir, 'trained_model')

    #load the model
    model = GRUModel(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_prob).to(device)
    model.load_state_dict(torch.load(model_path))
    
    #data loaders
    path = args.data_cache_dir

    test_set = eICU_Loader(path + 'test')
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=False)
    
    model.to(device)
    model.eval()

    running_loss = 0. 
    running_metrics = [0., 0., 0., 0., 0.]
    
    metrics = {"final_test_MAE": 0, "final_test_MAPE": 0, "final_test_MSE": 0, "final_test_MSLE": 0, "final_test_R_sq": 0}
    criterion = MSLELoss().to(device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, target = batch
            inputs, target = inputs.to(device), target.to(device)

            target = target.view(-1,1)  ; target.requires_grad_(True) 
            target = target.float()

            #run model
            outputs = model(inputs.float())
            outputs = outputs.to(device)
            
            #backpropagation and gradient calculation
            loss = criterion(outputs, target) 
        
            # Gather data and report
            running_loss += loss.item()

            #Get all the metrics and not just loss
            batch_metrics = Metrics.diff_metrics(target.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            running_metrics = [*map(sum, zip(running_metrics, batch_metrics))]

        #calculate the epoch level metrics
        avg_metrics = [metric/(i+1) for metric in running_metrics]
        
        for i, m in enumerate(metrics.keys()):
            metrics[m] = avg_metrics[i]

        wandb.log(metrics)

        print(avg_metrics)
        print(f'Performance against test: {metrics}')
        store_test_metrics(model_dir, avg_metrics)

        return


