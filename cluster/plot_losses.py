import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from os.path import isfile, join, basename
import sys
import glob
import pickle

base_dir = sys.argv[1]

def plot_losses(f, exp_name):
    hist_path = glob.glob(join(f, '*_model', 'history_*.dat'))[0]
    with open(hist_path, 'rb') as file_pi:
        hist = pickle.load(file_pi)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    plt.plot(hist['loss'], label='Training')
    plt.plot(hist['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist['lr'])
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')

    plt.suptitle(exp_name + '/' + basename(f));
    plt.savefig(join(f, exp_name+ '_' + basename(f)+'_losses.png'), pad_inches=0.0, bbox_inches='tight')

for f in glob.glob(join(base_dir, 'train_*')):
    if len(glob.glob(join(f, '*losses.png'))) == 0:
        plot_losses(f, basename(base_dir))
