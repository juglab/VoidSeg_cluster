from os.path import isdir, exists, join, basename
import os
from shutil import copy as cp
from shutil import move as mv
import glob
import sys
from PyInquirer import prompt, Validator, ValidationError
import argparse as ap

import json


def train_path(config):
        l = glob.glob('/projects/juglab/StarVoid/train_data/**', recursive=True)
        if len(l) == 0:
            raise Exception("No training data available in /projects/juglab/StarVoid/train_data/")

        return l


def test_path(config):
        l = glob.glob('/projects/juglab/StarVoid/test_data/**', recursive=True)
        if len(l) == 0:
            raise Exception("No training data available in /projects/juglab/StarVoid/test_data/")

        return l


class ValExpName(Validator):
    def validate(self, document):
        names = glob.glob('/projects/juglab/StarVoid/outdata/*')
        names = [n.split('/')[-1] for n in names]

        if document.text in names:
            raise ValidationError(
                message='An experiment with this name already exists. Please choose another name.',
                cursor_position=len(document.text)
            )
            
            
class TrainFracValidator(Validator):
    def validate(self, document):
        values = document.text.split(',')
        for v in values:
            try:
                int_v = float(v)
                if int_v > 100:
                    raise ValidationError(
                        message='Enter a comma separated list of integers between 0 and 100.',
                        cursor_position=len(document.text)
                    )
            except ValueError:
                raise ValidationError(
                    message='Enter a list of integers between 0 and 100.',
                    cursor_position=len(document.text)
                )


# class TrainFracValidator(Validator):
#     def validate(self, document):
#         values = document.text.split(',')
#         for v in values:
#             try:
#                 int_v = int(v)
#                 if int_v < 1 or int_v > 100:
#                     raise ValidationError(
#                         message='Enter a comma separated list of integers between 1 and 100.',
#                         cursor_position=len(document.text)
#                     )
#             except ValueError:
#                 raise ValidationError(
#                     message='Enter a list of integers between 1 and 100.',
#                     cursor_position=len(document.text)
#                 )
                


def main():
    parser = ap.ArgumentParser(description="Finetuning cluster job setup script.")
    parser.add_argument("--exp")
    parser.add_argument("--net")
    args, leftovers = parser.parse_known_args()

    if (args.exp is not None) and (args.net is not None):
        with open(args.exp) as f:
            exp_conf = json.load(f)

        with open(args.net) as f:
            net_conf = json.load(f)

        start_experiment(exp_conf, net_conf, 'train_'+str(exp_conf['train_frac']))
    else:
        questions = [
            {
                'type': 'input',
                'name': 'exp_name',
                'message': 'Experiment name:',
                'validate': ValExpName
            },
            {
                'type': 'list',
                'name': 'train_path',
                'message': 'Training data path:',
                'choices': train_path
            },
            {
                'type': 'list',
                'name': 'test_path',
                'message': 'Test data path:',
                'choices': test_path
            },
            {
                'type': 'confirm',
                'message': 'Use data augmentation during training?',
                'name': 'augment',
                'default': True
            },
            {
                'type': 'confirm',
                'message': 'Use random seeding during training?',
                'name': 'is_seeding',
                'default': True
            },
            {
                'type': 'input',
                'name': 'random_seed',
                'message': 'Random seed for training',
                'default': '42',
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'train_frac',
                'message': 'Training data fractions in x%:',
                'default': '1,2,3,5,10,20,30,40,50,60,70,80,90,100',
                'validate': TrainFracValidator,
                'filter': lambda val: [float(x) for x in val.split(',')]
            },
            {
                'type': 'list',
                'name': 'n_dim',
                'message': 'n_dim:',
                'choices': ['2', '3'],
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'axes',
                'message': 'axes',
                'default': 'YXC'
            },
            {
                'type': 'input',
                'name': 'n_channel_in',
                'message': 'n_channel_in',
                'default': '1',
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'n_channel_out',
                'message': 'n_channel_out',
                'default': '4',
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'unet_n_depth',
                'message': 'unet_n_depth',
                'default': '2',
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'unet_kern_size',
                'message': 'unet_kern_size',
                'default': '3',
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'unet_n_first',
                'message': 'unet_n_first',
                'default': '32',
                'filter': lambda val: int(val)
            },
            {
                'type': 'list',
                'name': 'unet_last_activation',
                'message': 'unet_last_activation',
                'choices': ['linear', 'relu'],
            },
            {
                'type': 'input',
                'name': 'unet_input_shape',
                'message': 'unet_input_shape',
                'default': 'None, None, 1',
                'filter': lambda val: tuple([None if x=='None' else int(x) for x in val.split(', ')])
            },
            {
                'type': 'list',
                'name': 'train_loss',
                'message': 'train_loss',
                'choices': ['mse', 'mae']
            },
            {
                'type': 'input',
                'name': 'train_epochs',
                'message': 'train_epochs',
                'default': '200',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'train_steps_per_epoch',
                'message': 'train_steps_per_epoch',
                'default': '400',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'train_learning_rate',
                'message': 'train_learning_rate',
                'default': '0.0004',
                'validate': lambda val: float(val) > 0,
                'filter': lambda val: float(val)
            },
            {
                'type': 'input',
                'name': 'train_batch_size',
                'message': 'train_batch_size',
                'default': '128',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'list',
                'name': 'train_tensorboard',
                'message': 'train_tensorboard',
                'choices': ['True', 'False'],
                'filter': lambda val: val == 'True'
            },
            {
                'type': 'input',
                'name': 'train_checkpoint',
                'message': 'train_checkpoint',
                'default': 'weights_best.h5'
            },
            {
                'type': 'input',
                'name': 'train_reduce_lr',
                'message': 'train_reduce_lr',
                'default': 'factor: 0.5, patience: 10',
                'filter': lambda val: {tmp.split(':')[0].strip() : float(tmp.split(':')[1].strip()) for tmp in val.split(',')}
            },
            {
                'type': 'list',
                'name': 'batch_norm',
                'message': 'batch_norm',
                'choices': ['True', 'False'],
                'filter': lambda val: val == 'True'
            },
            {
                'type': 'input',
                'name': 'train_scheme',
                'message': 'train_scheme',
                'default': 'Noise2Void'
            },
            {
                'type': 'input',
                'name': 'n2v_num_pix',
                'message': 'n2v_num_pix',
                'default': '64',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'n2v_patch_shape',
                'message': 'n2v_patch_shape',
                'default': '64, 64',
                'filter': lambda val: tuple([int(x) for x in val.split(', ')])
            },
            {
                'type': 'list',
                'name': 'n2v_manipulator',
                'message': 'n2v_manipulator',
                'choices': ['identity', 'uniform_withCP', 'normal_withoutCP', 'normal_additive', 'normal_fitted']
            },
            {
                'type': 'input',
                'name': 'n2v_neighborhood_radius',
                'message': 'n2v_neighborhood_radius',
                'default': '5'
            },
            {
                'type': 'confirm',
                'message': 'use_denoising',
                'name': 'use_denoising',
                'default': False,
                'filter': lambda val: int(val)
            },
            {
                'type': 'list',
                'name': 'scheme',
                'message': 'scheme',
                'choices': ['scheme1', 'scheme2', 'scheme3']
            }
        ]

        config = prompt(questions)
        config['probabilistic'] = False
        config['unet_residual'] = False
        config['n2v_manipulator'] = 'identity'
        pwd = os.getcwd()
        for p in config['train_frac']:
            os.chdir(pwd)
            exp_conf = {
                'exp_name' : config['exp_name'],
                'scheme' : config['scheme'],
                'train_path': config['train_path'],
                'test_path': config['test_path'],
                'is_seeding': config['is_seeding'],
                'random_seed': config['random_seed'],
                'augment': config['augment'],
                'train_frac': p,
                'base_dir': join('../..', config['exp_name']+config['scheme'], 'train_'+str(p)),
                'model_name': config['exp_name'].split('_')[0] + '_model'
            }

            net_conf = {}
            for k in config.keys():
                if k not in exp_conf.keys():
                    net_conf[k] = config[k]

            start_experiment(exp_conf, net_conf, 'train_'+str(p))


def start_experiment(exp_conf, net_conf, run_dir):
    if isdir(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir, exp_conf['model_name'])):
        confirmation = prompt([
            {
                'type': 'confirm',
                'message': 'This experiment already exists. Scripts will be replaced by current version and results will be overwritten. Do you want to continue?',
                'name': 'continue',
                'default': False
            }
        ])
        if confirmation['continue']:
            run(exp_conf, net_conf, run_dir)
        else:
            print('Abort')
    else:
        os.makedirs(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'],run_dir, exp_conf['model_name']), mode=0o775)
       
        with open(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'],run_dir, 'experiment.json'), 'w') as file:
            json.dump(exp_conf, file)

        with open(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'],run_dir, exp_conf['model_name'],  'config.json'), 'w') as file:
            json.dump(net_conf, file)

        os.makedirs(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'starvoid'), mode=0o775)
        os.makedirs(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'utils'), mode=0o775)

        os.system('chmod -R 775 '+'../../outdata/'+exp_conf['exp_name']+exp_conf['scheme'])

        run(exp_conf, net_conf, run_dir)


def run(exp_conf, net_conf, run_dir):
    for f in glob.glob(join('scripts', 'starvoid', '*')):
        cp(f, join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'starvoid', basename(f)))

    for f in glob.glob(join('scripts', 'utils', '*')):
        cp(f, join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'utils', basename(f)))

    log_file = 'experiment.log'


    os.chdir(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir))
    print('Current directory:', os.getcwd())
    cmd = "sbatch --exclude=r02n01 -p gpu --gres=gpu:1 --mem-per-cpu 256000 -t 48:00:00 --export=ALL -J StarVoid -o "+log_file+" scripts/starvoid/start_job_finetune.sh"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
