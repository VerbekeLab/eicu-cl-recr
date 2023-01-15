# Import dependencies
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')
    
class GRUModel(nn.Module): 
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob): 
        super(GRUModel, self).__init__() 

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True) 

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.gru_dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm = nn.BatchNorm1d(output_dim)  
        self.relu = nn.ReLU()

    def init_hidden(self, bsize):
        return torch.nn.init.orthogonal_(torch.randn(self.num_layers, bsize, self.hidden_dim).requires_grad_()).float().to(device)

    def forward(self, inputs):

        # inputs are of the shape [batch_size, seq_length, features]
        self.hidden = self.init_hidden(inputs.size(0))

        # run gru and obtain output
        outputs, _ = self.gru(inputs, self.hidden.detach())       
            
        out = self.gru_dropout(outputs)
        out = self.fc(out[:, -1, :])

        print(len(inputs))

        if len(inputs) < 2:
            out = self.relu(out)
        else:    
            out = self.batch_norm(out)                
            out = self.relu(out)

        return out


