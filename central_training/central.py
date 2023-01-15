import wandb
import argparse

from .train import train
from model.utils import load_config, set_api_key, load_yaml_config
from model.test import test
from fedml.arguments import add_args, Arguments

# from model.utils import plot_res

if __name__=='__main__':
    cf_args = add_args()
    args = Arguments(cf_args)

    '''
    ------ The below parsing preserves the nested format of the arguments in .yml
    ------ e.g. args.gru.bsize instead of args.bsize

        # parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     "--yaml_config_file",
        #     "--cf",
        #     help="yaml configuration file",
        #     type=str,
        #     default="",
        # )
        # args = parser.parse_args()
        # args = load_config(args.yaml_config_file)

    '''
    #get the args from yaml as a dict for wandb
    # config = load_yaml_config(cf_args.yaml_config_file)
    
    #Overwrite argument to specify central training
    args.central_training = 1
    print(f' argument for central training {args.central_training}')

    set_api_key(args.wandb_key)
    wandb.init(project=args.central_wandb_project)
    wandb.config.update(args)

    # call the train function
    print('Training started')
    train(args)

    #Run against test
    print('Running model against test')
    test(args, central=True)