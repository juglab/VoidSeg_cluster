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
                float_v = float(v)
                if float_v < 0 or float_v > 100:
                    raise ValidationError(
                        message='Enter a comma separated list of floats between 0 and 100.',
                        cursor_position=len(document.text)
                    )
            except ValueError:
                raise ValidationError(
                    message='Enter a list of floats between 0 and 100.',
                    cursor_position=len(document.text)
                )


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
                'default': '0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,100.0',
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
                'name': 'n2v_train_epochs',
                'message': 'n2v_train_epochs',
                'default': '200',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'seg_train_epochs',
                'message': 'seg_train_epochs',
                'default': '200',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'ini_train_epochs',
                'message': 'ini_train_epochs',
                'default': '200',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'n2v_train_steps_per_epoch',
                'message': 'n2v_train_steps_per_epoch',
                'default': '400',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'seg_train_steps_per_epoch',
                'message': 'seg_train_steps_per_epoch',
                'default': '400',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'ini_train_steps_per_epoch',
                'message': 'ini_train_steps_per_epoch',
                'default': '400',
                'validate': lambda val: int(val) > 0,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': 'n2v_train_learning_rate',
                'message': 'n2v_train_learning_rate',
                'default': '0.0004',
                'validate': lambda val: float(val) > 0,
                'filter': lambda val: float(val)
            },
            {
                'type': 'input',
                'name': 'seg_train_learning_rate',
                'message': 'seg_train_learning_rate',
                'default': '0.0004',
                'validate': lambda val: float(val) > 0,
                'filter': lambda val: float(val)
            },
            {
                'type': 'input',
                'name': 'ini_train_learning_rate',
                'message': 'ini_train_learning_rate',
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
                'choices': ['uniform_withCP', 'identity', 'normal_withoutCP', 'normal_additive', 'normal_fitted']
            },
            {
                'type': 'input',
                'name': 'n2v_neighborhood_radius',
                'message': 'n2v_neighborhood_radius',
                'default': '5'
            },
            {
                'type': 'list',
                'name': 'scheme',
                'message': 'scheme',
                'choices': ['baseline', 'sequential', 'finetune', 'finetune_denoised', 'finetune_denoised_noisy']
            }
        ]

        config = prompt(questions)
        config['probabilistic'] = False
        config['unet_residual'] = False
        
        pwd = os.getcwd()
        for run_idx in [1,2,3,4,5,6,7,8]:
            for p in config['train_frac']:
                if config['is_seeding']:
                    os.chdir(pwd)
                    run_name = config['exp_name']+'_run'+str(run_idx)
                    exp_conf, n2v_net, ini_net, seg_net = create_configs(config, run_name, seed=run_idx, train_frac=p)
                else:
                    os.chdir(pwd)
                    exp_conf, n2v_net, ini_net, seg_net = create_configs(config, config['exp_name'], seed=config['random_seed'], train_frac=p)

                start_experiment(exp_conf, n2v_net, ini_net, seg_net, 'train_'+str(p))


def create_configs(config, run_name, seed, train_frac):
    exp_conf = {
        'exp_name' : run_name,
        'scheme' : config['scheme'],
        'train_path': config['train_path'],
        'test_path': config['test_path'],
        'is_seeding': config['is_seeding'],
        'random_seed': seed,
        'augment': config['augment'],
        'train_frac': train_frac,
        'base_dir': join('../..', config['exp_name']+config['scheme'], 'train_'+str(train_frac)),
        'model_name': config['exp_name'].split('_')[0] + '_model'
    }

    n2v_net = create_n2v_net_config(config)
    ini_net = create_ini_net_config(config)
    seg_net = create_seg_net_config(config)

    return exp_conf, n2v_net, ini_net, seg_net


def create_n2v_net_config(config):
    n2v_net = {
        'n_dim' : config['n_dim'],
        'axes' : config['axes'],
        'use_denoising': 1,
        'n2v_neighborhood_radius' : config['n2v_neighborhood_radius'],
        'n2v_manipulator' : config['n2v_manipulator'],
        'n2v_patch_shape' : config['n2v_patch_shape' ],
        'n2v_num_pix' : config['n2v_num_pix'],
        'batch_norm' : config['batch_norm'],
        'train_reduce_lr' : config['train_reduce_lr'],
        'train_checkpoint' : config['train_checkpoint'],
        'train_tensorboard' : config['train_tensorboard'],
        'train_batch_size' : config[ 'train_batch_size'],
        'train_learning_rate' : config['n2v_train_learning_rate'],
        'train_steps_per_epoch' : config['n2v_train_steps_per_epoch'],
        'train_epochs' : config['n2v_train_epochs'],
        'train_loss' : config['train_loss'],
        'unet_input_shape' : config['unet_input_shape'],
        'unet_last_activation' : config['unet_last_activation'],
        'unet_n_first' : config['unet_n_first'],
        'unet_kern_size' : config['unet_kern_size'],
        'unet_n_depth' : config['unet_n_depth'],
        'n_channel_out' : config['n_channel_out'],
        'n_channel_in' : config['n_channel_in']
    }

    return n2v_net


def create_ini_net_config(config):
    ini_net = {
        'n_dim' : config['n_dim'],
        'axes' : config['axes'],
        'use_denoising': 1,
        'n2v_neighborhood_radius' : config['n2v_neighborhood_radius'],
        'n2v_manipulator' : config['n2v_manipulator'],
        'n2v_patch_shape' : config['n2v_patch_shape' ],
        'n2v_num_pix' : config['n2v_num_pix'],
        'batch_norm' : config['batch_norm'],
        'train_reduce_lr' : config['train_reduce_lr'],
        'train_checkpoint' : config['train_checkpoint'],
        'train_tensorboard' : config['train_tensorboard'],
        'train_batch_size' : config[ 'train_batch_size'],
        'train_learning_rate' : config['ini_train_learning_rate'],
        'train_steps_per_epoch' : config['ini_train_steps_per_epoch'],
        'train_epochs' : config['ini_train_epochs'],
        'train_loss' : config['train_loss'],
        'unet_input_shape' : config['unet_input_shape'],
        'unet_last_activation' : config['unet_last_activation'],
        'unet_n_first' : config['unet_n_first'],
        'unet_kern_size' : config['unet_kern_size'],
        'unet_n_depth' : config['unet_n_depth'],
        'n_channel_out' : config['n_channel_out'],
        'n_channel_in' : config['n_channel_in']
    }
    return ini_net


def create_seg_net_config(config):
    seg_net = {
        'n_dim': config['n_dim'],
        'axes': config['axes'],
        'use_denoising': 0,
        'n2v_neighborhood_radius': config['n2v_neighborhood_radius'],
        'n2v_manipulator': 'identity',
        'n2v_patch_shape': config['n2v_patch_shape'],
        'n2v_num_pix': config['n2v_num_pix'],
        'batch_norm': config['batch_norm'],
        'train_reduce_lr': config['train_reduce_lr'],
        'train_checkpoint': config['train_checkpoint'],
        'train_tensorboard': config['train_tensorboard'],
        'train_batch_size': config['train_batch_size'],
        'train_learning_rate': config['seg_train_learning_rate'],
        'train_steps_per_epoch': config['seg_train_steps_per_epoch'],
        'train_epochs': config['seg_train_epochs'],
        'train_loss': config['train_loss'],
        'unet_input_shape': config['unet_input_shape'],
        'unet_last_activation': config['unet_last_activation'],
        'unet_n_first': config['unet_n_first'],
        'unet_kern_size': config['unet_kern_size'],
        'unet_n_depth': config['unet_n_depth'],
        'n_channel_out': config['n_channel_out'],
        'n_channel_in': config['n_channel_in'],
    }
    return seg_net


def copy_exp_conf(exp_conf, run_dir):
    dir = join('..', '..', 'outdata', exp_conf['exp_name'] + exp_conf['scheme'], run_dir)
    if not os.path.isdir(dir):
        os.makedirs(dir, mode=0o775)
    with open(join(dir, 'experiment.json'),
              'w') as file:
        json.dump(exp_conf, file)


def copy_scripts(exp_conf, run_dir):
    os.makedirs(join('../..', 'outdata', exp_conf['exp_name'] + exp_conf['scheme'], run_dir, 'scripts', 'starvoid'),
                mode=0o775)
    os.makedirs(join('../..', 'outdata', exp_conf['exp_name'] + exp_conf['scheme'], run_dir, 'scripts', 'utils'),
                mode=0o775)

    for f in glob.glob(join('scripts', 'starvoid', '*')):
        cp(f, join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'starvoid', basename(f)))

    for f in glob.glob(join('scripts', 'utils', '*')):
        print(f)
        cp(f, join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir,'scripts', 'utils', basename(f)))


def create_outdir(exp_conf):
    os.system('chmod -R 775 ' + '../../outdata/' + exp_conf['exp_name'] + exp_conf['scheme'])


def copy_net_conf(exp_conf, run_dir, net_conf, model_type):
    os.makedirs(join('../..', 'outdata', exp_conf['exp_name'] + exp_conf['scheme'], run_dir,
                     exp_conf['model_name'] + model_type), mode=0o775)

    with open(join('../..', 'outdata', exp_conf['exp_name'] + exp_conf['scheme'], run_dir,
                   exp_conf['model_name'] + model_type, 'config.json'), 'w') as file:
        json.dump(net_conf, file)


def start_experiment(exp_conf, n2v_conf, ini_conf, seg_conf, run_dir):
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
            run(exp_conf, n2v_conf, ini_conf, seg_conf, run_dir)
        else:
            print('Abort')
    else:
        
        if(exp_conf['scheme'] == 'finetune_denoised_noisy'):
            copy_scripts(exp_conf, run_dir)
            copy_exp_conf(exp_conf, run_dir)
            copy_net_conf(exp_conf, run_dir, ini_conf, '_init')
            copy_net_conf(exp_conf, run_dir, seg_conf, '_seg')
            create_outdir(exp_conf)
            run(exp_conf, run_dir)
        elif(exp_conf['scheme'] == 'finetune_denoised'):
            copy_scripts(exp_conf, run_dir)
            copy_exp_conf(exp_conf, run_dir)
            copy_net_conf(exp_conf, run_dir, n2v_conf, '_denoise')
            copy_net_conf(exp_conf, run_dir, ini_conf, '_init')
            copy_net_conf(exp_conf, run_dir, seg_conf, '_seg')
            create_outdir(exp_conf)
            run(exp_conf, run_dir)
        elif(exp_conf['scheme'] == 'finetune'):
            copy_scripts(exp_conf, run_dir)
            copy_exp_conf(exp_conf, run_dir)
            copy_net_conf(exp_conf, run_dir, ini_conf, '_init')
            copy_net_conf(exp_conf, run_dir, seg_conf, '_seg')
            create_outdir(exp_conf)
            run(exp_conf, run_dir)
        elif(exp_conf['scheme'] == 'sequential'):
            copy_scripts(exp_conf, run_dir)
            copy_exp_conf(exp_conf, run_dir)
            copy_net_conf(exp_conf, run_dir, n2v_conf, '_denoise')
            copy_net_conf(exp_conf, run_dir, seg_conf, '_seg')
            create_outdir(exp_conf)
            run(exp_conf, run_dir)
        elif(exp_conf['scheme'] == 'baseline'):
            copy_scripts(exp_conf, run_dir)
            copy_exp_conf(exp_conf, run_dir)
            copy_net_conf(exp_conf, run_dir, seg_conf, '_seg')
            create_outdir(exp_conf)
            run(exp_conf, run_dir)
        else:
            print('Unknown scheme: {}'.format(exp_conf['scheme']))
        


def run(exp_conf, run_dir):
    log_file = 'experiment.log'

    os.chdir(join('../..', 'outdata', exp_conf['exp_name']+exp_conf['scheme'], run_dir))
    print('Current directory:', os.getcwd())
    cmd = "sbatch --exclude=r02n01 -p gpu --gres=gpu:1 --mem-per-cpu 256000 -t 48:00:00 --export=ALL -J StarVoid -o "+log_file+" scripts/starvoid/start_job_starvoid_clean.sh"
    print(cmd)
    # os.system(cmd)


if __name__ == "__main__":
    main()
