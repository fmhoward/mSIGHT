import argparse
from trainer import Reg_Trainer, Cyc_Trainer, P2p_Trainer
import yaml
import torch

assert torch.cuda.is_available()

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/RegGan.yaml', help='Path to the config file.')
    parser.add_argument('--run', type=str, default='evaluate', help='Training or Evaluation.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'RegGan':
        trainer = Reg_Trainer(config)
    elif config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)

    if opts.run == 'train':
        trainer.train()
    elif opts.run == 'evaluate':
        trainer.evaluate(config['eval_input'], config['eval_save'])
    

###################################
if __name__ == '__main__':
    main()