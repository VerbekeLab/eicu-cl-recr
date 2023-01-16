# Import dependencies
import os
import torch
import yaml
import wandb
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# Define the dataset class
class eICU_Loader(Dataset):

    def __init__(self, data_path, client_idx=None):

        if client_idx is not None:
            ts = (pd.read_csv(data_path + '/timeseries.csv')[lambda x: x['hospitalid'] == client_idx])
            static = (pd.read_csv(data_path + '/flat.csv')[lambda x: x['hospitalid'] == client_idx])
            labels = (pd.read_csv(data_path + '/labels.csv')[lambda x: x['hospitalid'] == client_idx])

        else:
            ts = pd.read_csv(data_path + '/timeseries.csv')
            static = pd.read_csv(data_path + '/flat.csv')
            labels = pd.read_csv(data_path + '/labels.csv')

        for data in [ts, static, labels]:
            data = data.drop(columns=['hospitalid'])

        # fix duplicates -> this should be looked into as to why these exist
        static = static.drop_duplicates(subset='patient', keep='first').reset_index()
        static.drop(columns=['index'])
        labels = labels.drop_duplicates(subset='patient', keep='first').reset_index()
        labels.drop(columns=['index'])

        # fix gendeer in static -> this should be done elsewhere
        static.gender = static.gender.replace('0', 0.0, regex=True)
        static.gender = static.gender.replace('1', 1.0, regex=True)
        static.gender = static.gender.replace('1.0', 1.0, regex=True)
        static.gender = static.gender.replace('0.0', 0.0, regex=True)
        static.gender = static.gender.replace('Unknown', 0.5, regex=True)
        static.gender = static.gender.replace('Other', 0.5, regex=True)
        static.gender = static.gender.fillna(0.5)

        # merge the data
        data = ts.merge(static, on=['patient'], how='inner')

        # set patient as the index
        self.data = data.set_index('patient')
        self.labels = labels

        self.data = self.data.fillna(0)
            
    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, index):
        self.index = index

        # get the patient
        patient = self.labels.patient[index]
        
        # get the LoS
        y = self.labels.actualiculos[self.labels.patient == patient].values.flatten()

        # get the data
        x = self.data[self.data.index == patient]

        if x.shape[0] > 24:
            x = x[0:24][:]

        # x = x.drop(columns=['hospitalid', 'time'])
        x = x.drop(columns=['time'])
        x = x.values

        return x, y


#MSLE Loss and Metrics Class
class MSLELoss(nn.Module):
  def __init__(self):
      super().__init__()
      self.mse = nn.MSELoss()    
  def forward(self, true, pred):
      return self.mse(torch.log(true + 1), torch.log(pred + 1))


class Metrics():
  def labelling_outcomes():
    true_labels = 2
    pred_labels = 2
    return true_labels, pred_labels

    true_labels, pred_labels = labelling_outcomes()
    cohen_kappa = metrics.cohen_kappa_score(true_labels, pred_labels)


  def diff_metrics(true, pred):
    mae = metrics.mean_absolute_error(true, pred)
    mape = metrics.mean_absolute_percentage_error(true, pred) #might have to perform the same trick like Rocheteau as mape can be very big for small ytrue values.
    mse = metrics.mean_squared_error(true, pred)
    msle = metrics.mean_squared_log_error(true, pred)#(true+1, pred+1)
    R_sq = metrics.r2_score(true, pred)
    return [mae, mape, mse, msle, R_sq]

def load_icu_data(args, train_path, test_path):
    print(f'bsize is {args.batch_size}')

    client_num = args.client_num_in_total
    class_num = args.output_dim

    #get the datasets
    train_set = eICU_Loader(train_path)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=False)
    
    test_set = eICU_Loader(test_path) #this is not the hold out test set
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=False)

    print('Loaded Global dataloaders')

    train_data_num = len(train_set)
    test_data_num = len(test_set)

    print(train_data_num, test_data_num)

    train_data_global = train_loader
    test_data_global = test_loader
    

    if not os.path.exists(args.data_cache_dir + 'hosp_dicts'):
        os.mkdir(args.data_cache_dir + 'hosp_dicts')

        #Local data is a dict
        # this is deprecated: train_data_local_num_dict 
        train_data_local_num_dict = {}
        train_data_local_dict = {}
        test_data_local_dict = {}

        print('Generating hospital level data: ')
        for client_idx in range(args.client_num_in_total):
            #this assumes that the hospital id is index 0 - len(clients)
            #check bsize, local data needs to be sufficient
            local_train_set = train_set = eICU_Loader(train_path, client_idx)
            local_test_set = eICU_Loader(test_path, client_idx)

            local_train_loader = DataLoader(local_train_set, args.batch_size, shuffle=True, drop_last=False)
            local_test_loader = DataLoader(local_test_set, args.batch_size, shuffle=True, drop_last=False)

            train_data_local_num_dict[client_idx] = len(local_train_set)
            train_data_local_dict[client_idx] = local_train_loader
            test_data_local_dict[client_idx] = local_test_loader

            print(f'Generating hospital level data for hospital: {client_idx}/{args.client_num_in_total}')
        
        joblib.dump(train_data_local_num_dict, args.data_cache_dir + 'hosp_dicts/local_num')
        joblib.dump(train_data_local_dict, args.data_cache_dir + 'hosp_dicts/local_train')
        joblib.dump(test_data_local_dict, args.data_cache_dir + 'hosp_dicts/local_test')

    else:
        print('Loading previously stored local training data')

        train_data_local_num_dict = joblib.load(args.data_cache_dir + 'hosp_dicts/local_num')
        train_data_local_dict = joblib.load(args.data_cache_dir + 'hosp_dicts/local_train')
        test_data_local_dict = joblib.load(args.data_cache_dir + 'hosp_dicts/local_test')

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

def load_data_federated(args):

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_icu_data(
        args,
        train_path=args.data_cache_dir + "train", #this needs to be updated to train
        test_path=args.data_cache_dir + "val",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset

def save_model(args, model, timestamp):
    stamp = 'training_{}'.format(timestamp)

    if args.central_training == 1:
        dir_path = os.path.join(args.central_model_dir, stamp)
    else:
        dir_path = os.path.join(args.dist_model_dir, stamp)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # arg_path = os.path.join(dir_path + '/' + 'args.json')

    # if not os.path.exists(arg_path):
    #     with open(arg_path, 'w') as fp:
    #         json.dump(args, fp, sort_keys = True, indent = 4)

    model_p = 'trained_model'
    torch.save(model.state_dict(), dir_path +'/'+ model_p) 


def store_metrics(args, timestamp, train_loss, train_metrics, validation_loss, validation_metrics):
    stamp = 'training_{}'.format(timestamp)
    
    if args.central_training == 1:
        dir_path = os.path.join(args.central_model_dir, stamp)
    else:
        dir_path = os.path.join(args.dist_model_dir, stamp)

    train_loss_df = pd.DataFrame(train_loss, columns=['train_loss']) ; train_loss_df.to_csv(dir_path + '/train_losses.csv', index=False)
    validation_loss_df = pd.DataFrame(validation_loss, columns=['validation_loss']); validation_loss_df.to_csv(dir_path + '/validation_losses.csv', index=False)
    
    train_metrics_df = pd.DataFrame(train_metrics, columns = ['MAE', 'MAPE', 'MSE', 'MSLE', 'R_sq']) ; train_metrics_df.to_csv(dir_path + '/train_metrics.csv', index=False)
    validation_metrics_df = pd.DataFrame(validation_metrics, columns = ['MAE', 'MAPE', 'MSE', 'MSLE', 'R_sq']) ; validation_metrics_df.to_csv(dir_path + '/validation_metrics.csv', index= False)

def store_test_metrics(dir_path, test_metrics):
    test_metrics_df = pd.DataFrame([test_metrics], columns = ['MAE', 'MAPE', 'MSE', 'MSLE', 'R_sq']) ; test_metrics_df.to_csv(dir_path + '/test_metrics.csv', index=False)
   

#this parser respects nested format -- still need to parse the --cf path from cli input and provide as yaml_path
def load_config(yaml_path):
    args = load_yaml_config(yaml_path)
    args = obj(args)
    return args

def load_yaml_config(yaml_path):
    """Helper function to load a yaml config file"""
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")

class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)


#wandb config
def set_api_key(api_key = None):
    WANDB_ENV_VAR = "WANDB_API_KEY"
    if api_key:
        os.environ[WANDB_ENV_VAR] = api_key
    elif not os.environ.get(WANDB_ENV_VAR):
        try:
            # Check if user is already logged into wandb.
            wandb.ensure_configured()
            if wandb.api.api_key:
                print("Already logged into W&B.")
                return
        except AttributeError:
            pass
        raise ValueError(
            "No WandB API key found. Either set the {} environment "
            "variable, pass `api_key` or `api_key_file` to the"
            "`WandbLoggerCallback` class as arguments, "
            "or run `wandb login` from the command line".format(WANDB_ENV_VAR)
        )


if __name__=='__main__':
    val_set = eICU_Loader('/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_Data/val/')
    val_loader = DataLoader(val_set, 2, True, drop_last=True)
    for i, batch in enumerate(val_loader):
        input, target = batch
        print(input.dtype)
        print(target.dtype)

        if i == 2:
            break


