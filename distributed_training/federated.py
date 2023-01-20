import fedml
from fedml import FedMLRunner

from model.model import GRUModel
from model.utils import load_data_federated
from model.test import test
from distributed_training.recruitment import cl_recruitment, update_dataset_args

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data_federated(args)

    if args.recruitment:
        client_list = cl_recruitment(args, dataset)
    #     args, dataset = update_dataset_args(args, client_list, dataset)

    # # load model (the size of MNIST image is 28 x 28)
    # model = GRUModel(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_prob)
    
    # # start training
    # simulator = FedMLRunner(args, device, dataset, model)
    # simulator.run()

    # #run against test
    # test(args, central=False)