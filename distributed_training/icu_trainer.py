
import torch
from torch import nn

from fedml.core.alg_frame.client_trainer import ClientTrainer
from model.utils import MSLELoss, Metrics


class ModelTrainerICU(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = MSLELoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        elif args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        elif args.client_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for i, batch in enumerate(train_data):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)

                target = target.view(-1,1)  ; target.requires_grad_(True) 
                target = target.float()
                    
                #zero grad optimizer for every batch
                model.zero_grad()
                optimizer.zero_grad()

                #run model
                outputs = model(inputs.float())
                outputs = outputs.to(device)
                
                #backpropagation and gradient calculation
                loss = criterion(outputs, target) 
                loss.backward()

                #Adjust model weights
                optimizer.step()        
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        running_loss = 0. 
        running_metrics = [0., 0., 0., 0., 0.]
        
        metrics = {"MAE": 0, "MAPE": 0, "MSE": 0, "MSLE": 0, "R_sq": 0}
        criterion = MSLELoss().to(device)

        with torch.no_grad():
            for i, batch in enumerate(test_data):
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

            return metrics