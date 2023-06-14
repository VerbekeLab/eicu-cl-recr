import fedml
import numpy as np
import pandas as pd
from fedml import FedMLRunner

from model.model import GRUModel, LSTMModel
from model.utils import load_data_federated
from model.test import test
from distributed_training.recruitment import cl_recruitment, update_dataset_args

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    if not args.client_selection_seed:
        rng = np.random.default_rng(seed=None)
        rseed = int(rng.integers(123456789, size=1))
        args.random_seed = rseed

    
    if args.experiment:
    # paremeter settings for div and samplesize experiments
        if args.div_exp:
            params = pd.read_csv('/data/leuven/345/vsc34578/eicu-cl-req/recr_params_div.csv')
            div = round(params.divergence[0], 2)
            sam = round(params.samplesize[0], 2)
            args.div_weight = div 
            args.inst_weight = sam
            print(f'divergence setting: {args.div_weight} and sample setting {args.inst_weight}')
            params.divergence[0] = div + 0.05
            params.samplesize[0] = sam - 0.05
            params.to_csv('/data/leuven/345/vsc34578/eicu-cl-req/recr_params_div.csv', index = False)
        else:
            params = pd.read_csv('/data/leuven/345/vsc34578/eicu-cl-req/recr_params_sam.csv')
            div = round(params.divergence[0], 2)
            sam = round(params.samplesize[0], 2)
            args.div_weight = div 
            args.inst_weight = sam
            print(f'divergence setting: {args.div_weight} and sample setting {args.inst_weight}')
            params.divergence[0] = div - 0.05
            params.samplesize[0] = sam + 0.05
            params.to_csv('/data/leuven/345/vsc34578/eicu-cl-req/recr_params_sam.csv', index = False)
    
    
    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data_federated(args)

    if args.recruitment:
        client_list = cl_recruitment(args, dataset)
        args, dataset = update_dataset_args(args, client_list, dataset)

    # load model
    if args.model == 'GRU':
        model = GRUModel(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_prob).to(device)
    elif args.model == 'LSTM':
        model = LSTMModel(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_prob).to(device)
    
    # start training
    simulator = FedMLRunner(args, device, dataset, model)
    simulator.run()

    #run against test
    test(args, central=False)