
import time
import torch
from datetime import datetime

from torch.utils.data import DataLoader

from model.model import GRUModel

from model.utils import MSLELoss
from model.utils import Metrics
from model.utils import eICU_Loader

from model.utils import save_model, store_metrics

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model
def train(args):
    #set vars to track loss and metrics
    train_loss = [] ; train_metrics = []
    validation_loss = [] ; validation_metrics = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    #data loaders
    path = args.get('path')

    train_set = eICU_Loader(path + 'train') #this is just for local testing ############################ update to train
    train_loader = DataLoader(train_set, args.get('bsize'), shuffle=True, drop_last=True)
    
    valid_set = eICU_Loader(path + 'val')
    valid_loader = DataLoader(valid_set, args.get('bsize'), shuffle=True, drop_last=True)

    #model, optimizer and loss
    model = GRUModel(args.get('input_dim'), args.get('hidden_dim'), args.get('num_layers'), args.get('output_dim'), args.get('dropout_prob')).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.get('lr') , weight_decay=args.get('wd'))
    loss_fn = MSLELoss()

    for epoch in range(args.get('epochs')):
        #track epoch level time
        start_time = time.time()

        model.train(True)
        t_loss, t_metrics = train_epoch(model, loss_fn, optimizer, train_loader, epoch+1)

        model.train(False)
        v_loss, v_metrics = validate_epoch(model, loss_fn, valid_loader, epoch+1)

        #print epoch level metrics
        print('Epoch {} train metrics: {}'.format(epoch+1, t_metrics))
        print('Epoch {} validation metrics: {}'.format(epoch+1, v_metrics)) 
    
        #store metrics
        train_loss.append(t_loss) ; train_metrics.append(t_metrics)
        validation_loss.append(v_loss); validation_metrics.append(v_metrics)

        #store epoch level model
        save_model(args, model, timestamp, epoch+1)

        #track time
        print('Training for epoch {} took: {}'.format(epoch+1, time.time() - start_time))

        #TODO
        # -- check the early stopping criteria 
        # -- log to wandb
    
    #write to csv -> better to do this per epoch not to lose information when training is interupted
    store_metrics(args, timestamp, train_loss, train_metrics, validation_loss, validation_metrics)

    #TODO
    # -- run against test


def train_epoch(model, loss_fn, optimizer, data_loader, epoch_n):
    #track the metrics
    last_loss = 0. ; running_loss = 0. 
    running_metrics = [0., 0., 0., 0., 0.]
    
    for i, batch in enumerate(data_loader):
        inputs, target = batch
        inputs, target = inputs.to(device), target.to(device)

        target = target.view(-1,1)  ; target.requires_grad_(True) 
        target = target.float()
            
        #zero grad optimizer for every batch
        optimizer.zero_grad()

        #run model
        outputs = model(inputs.float())
        outputs = outputs.to(device)
        
        #backpropagation and gradient calculation
        loss = loss_fn(outputs, target) 
        loss.backward()

        #Adjust model weights
        optimizer.step()        
        
        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss/(i+1)

        #Get all the metrics and not just loss
        batch_metrics = Metrics.diff_metrics(target.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        running_metrics = [*map(sum, zip(running_metrics, batch_metrics))]
        
        #print some information to stdout
        print('Epoch {}, batch {} ------ loss: {} ------ average loss: {}.'.format(epoch_n, i + 1, loss.item(), last_loss)) 

    #calculate the epoch level metrics
    epoch_metrics = [metric/(i+1) for metric in running_metrics]
    
    return last_loss, epoch_metrics

def validate_epoch(model, loss_fn, data_loader, epoch_n):
    last_loss = 0. ; running_loss = 0. 
    running_metrics = [0., 0., 0., 0., 0.]

    for i, batch in enumerate(data_loader):
        inputs, target = batch
        inputs, target = inputs.to(device), target.to(device)

        target = target.view(-1,1)  ; target.requires_grad_(True) 
        target = target.float()

        #run model
        outputs = model(inputs.float())
        outputs = outputs.to(device)
        
        #backpropagation and gradient calculation
        loss = loss_fn(outputs, target) 
       
        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss/(i+1)

        #Get all the metrics and not just loss
        batch_metrics = Metrics.diff_metrics(target.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        running_metrics = [*map(sum, zip(running_metrics, batch_metrics))]
        
        #print some information to stdout
        print('Epoch {}, batch {} ------ val_loss: {} ------ average val_loss: {}.'.format(epoch_n, i + 1, loss.item(), last_loss)) 

    #calculate the epoch level metrics
    epoch_metrics = [metric/(i+1) for metric in running_metrics]
    
    return last_loss, epoch_metrics