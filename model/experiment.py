import json
import argparse

from model.train import train
# from model.test import test

# from model.utils import plot_res


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--argpath', nargs=1,
                        help="JSON file to be processed", type=str)
    args = parser.parse_args()

    with open(args.argpath[0], 'r') as f:
        parameters = json.load(f)
    
    # call the train function
    print('Training started')
    train(parameters)