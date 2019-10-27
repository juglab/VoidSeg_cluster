# VoidSeg Documentation (27.10.2019)
Under review at ISBI 2020. 

Deep learning (DL) has arguably emerged as the method of choice for the detection and segmentation of biological structures in microscopy images. 
However, DL typically needs copious amounts of annotated training data that is for biomedical projects typically not available and excessively expensive to generate. 
Additionally, tasks become harder in the presence of noise, requiring even more high-quality training data.
Hence, we propose to use denoising networks to improve the performance of other DL-based image segmentation methods. 
More specifically, we present ideas on how state-of-the-art self-supervised CARE networks can improve cell/nuclei segmentation in microscopy data. 
Using two state-of-the-art baseline methods, U-Net and StarDist, we show that our ideas consistently improve the quality of resulting segmentations, especially when only limited training data for noisy micrographs are available.
An overview of the Noise2Seg (StarVoid/VoidSeg) project. This branch contains the working implementation of VoidSeg which is used for the current research regarding the question if N2V training could help the training of a 3-class segmentation network. 

## Datasets used
All data used for the experiments can be found at https://owncloud.mpi-cbg.de/index.php/s/P4qDOyJEnsaWK55
The link contains two folders namely- `DSB` and `BBBC`. Inside each folder, there are three subfolders named (i)`train_data` containing training and validation data, (ii) `test_data` containing test data, and (iii) `seg_gt` containing ground truth instance segmentations for test data.
For DSB dataset, we use noise levels n10, n20 and n40 (where n10 represents corruption of clean (original) microscopy images with Gaussian noise of mean 0 and std 10). For BBBC dataset, we use noise levels n150 and n200. 

For example, the training and validation data for noise level n10 for DSB dataset is located in `DSB/train_data/DSB2018_TrainVal10.npz`. 

Each of the .npz files in sub-folder train_data has four keys- `X_train`, `Y_train`, `X_val` and `Y_val`.
Each of the .npz files in sub-folder test_data has two keys- `X_test`, `Y_test`.

## Experiments to run
The following experiments can be run with the code.
* U-Net baseline 
* U-Net Sequential
* U-Net finetune
* U-Net finetune sequential
* StarDist baseline
* StarDist Sequential

All the experiments listed above can be divided intow two module: denoising and segmentation. For U-Net base;line and StarDist, we only need segmentation modules while for all other schemes both denoising and segmentation modules are needed. We provide scripts to run both these tasks in a modular manner.

## Start an Experiment
We created multiple scripts to setup experiments which will trigger individual cluster-jobs on falcon. 
There are different `run_EXPERIMENT.py` scripts located in `cluster/`:
Before running any of the scripts, make sure to create `outdata`, `train_data`, `test_data` directories. All data used for training and testing will be loaded from `train_data` and `test_data` respectively and all results are written to `outdata`.
The directory strcture should look like the following. 
`outdata`
`train_data`
`test_data`
`StarVoid/cluster/` (This is created just by cloning this repository). This means that the path `../../StarVoid/cluster`  points to a directory where `outdata`, `train_data` and `test_data` should be. After getting the directory structure, we are ready to run experiments. 

---------------------------------------------Denoising with N2V---------------------------------------------------------------
For running denoising with U-Net, run the script `run_starvoid.py` from `StarVoid/cluster` . To run this, simply use `python3 run_starvoid.py`. This will bring up a commad line interface displayin some parameters that need to be entered for training a N2V network. The following parameters are asked:
* Experiment name: This is the name of the directory in `falcon:/projects/juglab/StarVoid/outdata/` to which all outputs of this experiment are written.
* Scheme: Two options: denoising and segmentation come up. Choose denoising for running N2V.
* Training data path: This will recursivly list all files in `train_data` and the user has to select the file containing all (100%) of the training data. (Training data has to be pre-prepared)
* Test data path: Same for the test-data.
* Use data augmentation during training
* Random seed for training: This parameter asks for a seed to seed numpy.random. This allows for consistent training-data shuffling.
* Training data fractions in x%: For each fraction a individual experiment is started and only x% of the training data are used. For each fraction a directory in `outdata/EXPERIMENT_NAME/` is created. Choose `100%` only for denoising as we want to use all noisy data for unsupervised denoising training.
* Now a list of network architecture and training scheme parameters follow. For example, parameters like `number_of_epochs`, `steps_per_epoch`, `path_to_train_data`, `path_to_test_data`, `batch_size`, `depth`, etc. will be asked. We have set the parameters we used in the paper as default.

After the denoising is run, the script automatically creates a folder `outdata/EXPERIMENT_NAME/` and saves the best weights of N2V network in `outdata/EXPERIMENT_NAME/train_100.0/`. Also the denoised train data and test data are saved in `train_data` and `test_data` folders with name `N2V_EXPERIMENT_NAME_TrainVal.npz` and `N2V_EXPERIMENT_NAME_Test.npz`.

---------------------------------------------Segmentation with U-Net----------------------------------------------------------
Next, to run U-Net segmentation, again run `run_starvoid.py` as above but selecting `segmentation` for `Scheme` on command line interface. Choose appropriate parameters as described above.
Similar to denoiisng, for each experiment&training-fraction a directory `outdata/EXPERIMENT_NAME/train_X%` is created with the following directories/files:
* `scripts/`: This directory contains a copy of all scripts which are needed to run the experiment. In fact these are the scripts which got executed to produce the results.
* `XX_model/`: This directory contains the network configuration, training history, last_weights.h5, best_weights.h5, and the tensorboard-events (if activated). XX is replaced by the part of the EXPERIMENT_NAME which comes before the first _ corresponding to the network-type.
* `experiment.json`: This is the experiment config file which is written by the `run_EXPERIMENT.py`.
* `experiment.log`: The log of this run.
* `seg_scores.dat`: Contains the all seg-scores computed on the computed outputs for different threshold.
* `best_scores.dat`: Contains the best seg-scores.
* `backgroundXXX.tif`: Is the background segmentation result for the best average precision/seg-score.
* `borderXXX.tif`: Is the border segmentation result for the best average precision/seg-score.
* `foregroundXXX.tif`: Is the foreground segmentation result for the best average precision/seg-score. 
* `maskXXX.tif`: Is the instance segmentation result for the best average precision/seg-score.

Using the segmentation procedure described above, U-Net baseline, U-Net sequential, U-Net finetune and U-Net Finetune sequential schemes can be run. It should be noted that for U-Net Sequential, U-Net finetune and U-Net Finetune sequential schemes, segmentation part must be performed after running denoising part in the manner described above. Just make sure to choose the right denoised train and test data when prompted by the script on the command line for U-Net Sequential and U-Net Finetune sequential schemes. Also, for U-Net Finetune and U-Net Finetune sequential schemes, the command line interface will prompt to enter the path to the weights of N2V network for initialization of segmentation network. Please select the appropriate path.

---------------------------------------------Segmentation with StarDist-------------------------------------------------------
Running segmentation with StarDist is similar to segmentation with U-Net but using the script `run_stardist.py` and choosing parameters from command line interface when prompted as described above. For running StarDist Sequential, first run denosing as described above, then run StarDist choosing the denoised train and test data when prompted by the command line interface.

## Experiment Evaluation
To evaluate and compare results of multiple experiments we have two scripts:

* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_avg_precision_scores.py`:
    This script will plot the average best seg-scores of multiple runs over all available train-data fractions as a line-plot. To call this script move to `falcon:/projects/juglab/StarVoid/outdat/` and call `python3 ../StarVoid/cluster/plot_seg_scores.py -config avg_plot.json`. The provied `avg_plot.json` contains a dictionary with the following parameters:
    - `"exp_name" : ["exp_name1", "exp_name2"]` -> List of the averaging experiments. These names have to appear again in this json-file defining which experiment-runs have to be averaged.
    - `"exp_name1" : ["EXPERIMENTNAME_run0", "EXPERIMENTNAME_run1"]` -> List of all runs that have to be averaged. 
    - `"gt" : "../path/to/gt/"`
    - `"output" : "name_of_plot.png"`
* `falcon:/projects/juglab/StarVoid/StarVoid/cluster/plot_avg_seg_scores.py`: This is similar to plotting average precision scores as described above with the evaluation metric used being SEG.

