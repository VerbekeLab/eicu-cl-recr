# Import dependencies
import os
import torch
import json
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class eICU_Loader(Dataset):

    def __init__(self, data_path):
        
        # read in the data
        ts = pd.read_csv(data_path + '/timeseries.csv')
        static = pd.read_csv(data_path + '/flat.csv')
        labels = pd.read_csv(data_path + '/labels.csv')

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
        print(x)
        print(x.dtype)
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


def save_model(args, model, timestamp, epoch):
    stamp = 'training_{}'.format(timestamp)
    dir_path = os.path.join(args.get('path'), stamp)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    arg_path = os.path.join(dir_path + '/' + 'args.json')

    if not os.path.exists(arg_path):
        with open(arg_path, 'w') as fp:
            json.dump(args, fp, sort_keys = True, indent = 4)

    model_p = 'model_epoch_{}'.format(epoch)
    torch.save(model.state_dict(), dir_path +'/'+ model_p) 


def store_metrics(args, timestamp, train_loss, train_metrics, validation_loss, validation_metrics):
    stamp = 'training_{}'.format(timestamp)
    dir_path = os.path.join(args.get('path'), stamp)

    train_loss_df = pd.DataFrame(train_loss, columns=['train_loss']) ; train_loss_df.to_csv(dir_path + '/train_losses.csv', index=False)
    validation_loss_df = pd.DataFrame(validation_loss, columns=['validation_loss']); validation_loss_df.to_csv(dir_path + '/validation_losses.csv', index=False)
    
    train_metrics_df = pd.DataFrame(train_metrics, columns = ['MAE', 'MAPE', 'MSE', 'MSLE', 'R_sq']) ; train_metrics_df.to_csv(dir_path + '/train_metrics.csv', index=False)
    validation_metrics_df = pd.DataFrame(validation_metrics, columns = ['MAE', 'MAPE', 'MSE', 'MSLE', 'R_sq']) ; validation_metrics_df.to_csv(dir_path + '/validation_metrics.csv', index= False)


if __name__=='__main__':
    val_set = eICU_Loader('/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_Data/val/')
    val_loader = DataLoader(val_set, 2, True, drop_last=True)
    for i, batch in enumerate(val_loader):
        input, target = batch
        print(input.dtype)
        print(target.dtype)

        if i == 2:
            break
