# Noise2Seg Documentation (27.05.2019)
An overview of the Noise2Seg (StarVoid/VoidSeg) project. This branch contains the working implementation of VoidSeg which is used for the current research regarding the question if N2V training could help the training of a 3-class segmentation network. 

__Note:__ The relative paths are currently hardcoded!

We created multiple scripts to setup experiments which will trigger individual cluster-jobs on falcon. The current developer directory is:
`falcon:/projects/juglab/StarVoid`

In there we have `StarVoid`, `StarVoid_alex`, `StarVoid_b`, `StarVoid_M`, `StarVoid_Manan` all of these are clones of this Git-Repository with different owners. Unfortunately we didn't found a better way to work with the owner-ship rights of the git-repository. 

Additional important directories are `train_data` and `outdata`. All data used for training is loaded from `train_data` and all results are written to `outdata`.

## Start an Experiment
There are different `run_EXPERIMENT.py` scripts located in `falcon:/projects/juglab/StarVoid/StarVoid/cluster/`:
* `run_finetune.py`:
    This will start an experiment which will load a pre-trained network and fine-tune it.
* `run_n2v.py`:
    This will train standard denoising networks with the n2v training scheme.
* `run_stardist.py`:
    This will train standard stardist networks.
* `run_starvoid.py`:
    This will train segmentation networks. The last parameter `use_denoising` can turn denoising on or off. If `use_denoising=0` only the segmentation loss is computed. If `use_denoising=1` the segmentation and denoising loss are computed jointly. 
   
Each of these can be started via `python3 run_EXPERIMENT.py`. The following parameters are asked:
* Experiment name: This is the name of the directory in `falcon:/projects/juglab/StarVoid/outdata/` to which all outputs of this experiment are written. We use the following naming convention: network_dataset_noise_additionalInfo_runX with:
  - network in `{sd == stardist, su == sequential UNet aka W-Net, sv == starvoid}`
  - dataset in `{DSB, CTC}`
  - noise in `{n0, n20, n40}` corresponds to the train_data-set
  - runX is used to indicate differently seeded runs of the same experiment. We usually use X as seed. 
* Training data path: This will recursivly list all files in `falcon:/projects/juglab/StarVoid/train_data` and the user has to select the file containing all (100%) of the training data. (Training data has to be pre-prepared)
* Test data path: Same for the test-data.
* Use data augmentation during training
* Random seed for training: This parameter asks for a seed to seed numpy.random. This allows for consistent training-data shuffling.
* Training data fractions in x%: For each fraction a individual experiment is started and only x% of the training data are used. For each fraction a directory in `outdata/EXPERIMENT_NAME/` is created. 
* Now a list of network architecture and training scheme parameters follow.

For each experiment&training-fraction a directory `outdata/EXPERIMENT_NAME/train_X%` is created with the following directories/files:
* `scripts/`: This directory contains a copy of all scripts which are needed to run the experiment. In fact these are the scripts which got executed to produce the results.
* `XX_model/`: This directory contains the network configuration, training history, last_weights.h5, best_weights.h5, and the tensorboard-events (if activated). XX is replaced by the part of the EXPERIMENT_NAME which comes before the first _ corresponding to the network-type.
* `experiment.json`: This is the experiment config file which is written by the `run_EXPERIMENT.py`.
* `experiment.log`: The log of this run.
* `seg_scores.dat`: Contains the all seg-scores computed on the computed outputs for different threshold.
* `best_scores.dat`: Contains the best seg-scores.
* `backgroundXXX.tif`: Is the background segmentation result for the best seg-score.
* `borderXXX.tif`: Is the border segmentation result for the best seg-score.
* `foregroundXXX.tif`: Is the foreground segmentation result for the best seg-score. 

## Experiment Evaluation
To evaluate and compare results of multiple experiments we have two scripts:
* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_seg_scores.py`:
    This script will plot the best seg-scores over all available train-data fractions as a line-plot. To call this script move to `falcon:/projects/juglab/StarVoid/outdat/` and call `python3 ../StarVoid/cluster/plot_seg_scores.py -g path/to/gt/seg-data -out name_of_the_plot.png experiment_name1 experiment_name2` the last n parameters are all the experiments which should be plotted.
* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_avg_seg_scores.py`:
    This script will plot the average best seg-scores of multiple runs over all available train-data fractions as a line-plot. To call this script move to `falcon:/projects/juglab/StarVoid/outdat/` and call `python3 ../StarVoid/cluster/plot_seg_scores.py -config avg_plot.json`. The provied `avg_plot.json` contains a dictionary with the following parameters:
    - `"exp_name" : ["exp_name1", "exp_name2"]` -> List of the averaging experiments. These names have to appear again in this json-file defining which experiment-runs have to be averaged.
    - `"exp_name1" : ["EXPERIMENTNAME_run0", "EXPERIMENTNAME_run1"]` -> List of all runs that have to be averaged. 
    - `"gt" : "../path/to/gt/"`
    - `"output" : "name_of_plot.png"`
# Noise2Seg Documentation (27.05.2019)
An overview of the Noise2Seg (StarVoid/VoidSeg) project. This branch contains the working implementation of VoidSeg which is used for the current research regarding the question if N2V training could help the training of a 3-class segmentation network. 

__Note:__ The relative paths are currently hardcoded!

We created multiple scripts to setup experiments which will trigger individual cluster-jobs on falcon. The current developer directory is:
`falcon:/projects/juglab/StarVoid`

In there we have `StarVoid`, `StarVoid_alex`, `StarVoid_b`, `StarVoid_M`, `StarVoid_Manan` all of these are clones of this Git-Repository with different owners. Unfortunately we didn't found a better way to work with the owner-ship rights of the git-repository. 

Additional important directories are `train_data` and `outdata`. All data used for training is loaded from `train_data` and all results are written to `outdata`.

## Start an Experiment
There are different `run_EXPERIMENT.py` scripts located in `falcon:/projects/juglab/StarVoid/StarVoid/cluster/`:
* `run_finetune.py`:
    This will start an experiment which will load a pre-trained network and fine-tune it.
* `run_n2v.py`:
    This will train standard denoising networks with the n2v training scheme.
* `run_stardist.py`:
    This will train standard stardist networks.
* `run_starvoid.py`:
    This will train segmentation networks. The last parameter `use_denoising` can turn denoising on or off. If `use_denoising=0` only the segmentation loss is computed. If `use_denoising=1` the segmentation and denoising loss are computed jointly. 
   
Each of these can be started via `python3 run_EXPERIMENT.py`. The following parameters are asked:
* Experiment name: This is the name of the directory in `falcon:/projects/juglab/StarVoid/outdata/` to which all outputs of this experiment are written. We use the following naming convention: network_dataset_noise_additionalInfo_runX with:
  - network in `{sd == stardist, su == sequential UNet aka W-Net, sv == starvoid}`
  - dataset in `{DSB, CTC}`
  - noise in `{n0, n20, n40}` corresponds to the train_data-set
  - runX is used to indicate differently seeded runs of the same experiment. We usually use X as seed. 
* Training data path: This will recursivly list all files in `falcon:/projects/juglab/StarVoid/train_data` and the user has to select the file containing all (100%) of the training data. (Training data has to be pre-prepared)
* Test data path: Same for the test-data.
* Use data augmentation during training
* Random seed for training: This parameter asks for a seed to seed numpy.random. This allows for consistent training-data shuffling.
* Training data fractions in x%: For each fraction a individual experiment is started and only x% of the training data are used. For each fraction a directory in `outdata/EXPERIMENT_NAME/` is created. 
* Now a list of network architecture and training scheme parameters follow.

For each experiment&training-fraction a directory `outdata/EXPERIMENT_NAME/train_X%` is created with the following directories/files:
* `scripts/`: This directory contains a copy of all scripts which are needed to run the experiment. In fact these are the scripts which got executed to produce the results.
* `XX_model/`: This directory contains the network configuration, training history, last_weights.h5, best_weights.h5, and the tensorboard-events (if activated). XX is replaced by the part of the EXPERIMENT_NAME which comes before the first _ corresponding to the network-type.
* `experiment.json`: This is the experiment config file which is written by the `run_EXPERIMENT.py`.
* `experiment.log`: The log of this run.
* `seg_scores.dat`: Contains the all seg-scores computed on the computed outputs for different threshold.
* `best_scores.dat`: Contains the best seg-scores.
* `backgroundXXX.tif`: Is the background segmentation result for the best seg-score.
* `borderXXX.tif`: Is the border segmentation result for the best seg-score.
* `foregroundXXX.tif`: Is the foreground segmentation result for the best seg-score. 

## Experiment Evaluation
To evaluate and compare results of multiple experiments we have two scripts:
* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_seg_scores.py`:
    This script will plot the best seg-scores over all available train-data fractions as a line-plot. To call this script move to `falcon:/projects/juglab/StarVoid/outdat/` and call `python3 ../StarVoid/cluster/plot_seg_scores.py -g path/to/gt/seg-data -out name_of_the_plot.png experiment_name1 experiment_name2` the last n parameters are all the experiments which should be plotted.
* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_avg_seg_scores.py`:
    This script will plot the average best seg-scores of multiple runs over all available train-data fractions as a line-plot. To call this script move to `falcon:/projects/juglab/StarVoid/outdat/` and call `python3 ../StarVoid/cluster/plot_seg_scores.py -config avg_plot.json`. The provied `avg_plot.json` contains a dictionary with the following parameters:
    - `"exp_name" : ["exp_name1", "exp_name2"]` -> List of the averaging experiments. These names have to appear again in this json-file defining which experiment-runs have to be averaged.
    - `"exp_name1" : ["EXPERIMENTNAME_run0", "EXPERIMENTNAME_run1"]` -> List of all runs that have to be averaged. 
    - `"gt" : "../path/to/gt/"`
    - `"output" : "name_of_plot.png"`
    
## Finetuning schemes

In  order to improve segmentation, we decided to initialize our segmentation networks (both U-Net and W-Net) with Noise2Void trained denoising weights. Then the segmentation network is trained on segmentation loss under the following schemes:
For U-Net:

Scheme 1: Retraining all weights of segmentation network with n2v initialization. 
Scheme 2: Freezing the weights of the downsampling (encoder) part of the U-Net and the retraining only upsampling (decoder) part.
Scheme 3: Freezing all weights except the last layer
Scheme 4: Freezing all weights except the last layer for 10 epochs and then unfreezing all and retraining for 90 more epochs

For W-Net:

Coming soon...

