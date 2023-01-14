import fedml
from fedml import FedMLRunner

from model.model import GRUModel
from model.utils import load_data_federated
from model.test import test

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data_federated(args)

    # load model (the size of MNIST image is 28 x 28)
    model = GRUModel(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_prob)
    print('reached target point: model and data loaded -----> start training federated')
    
    # start training
    simulator = FedMLRunner(args, device, dataset, model)
    simulator.run()

    #run against test
    test(args, central=False)