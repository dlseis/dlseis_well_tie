# Wavelet Extraction (deconvolution) with Deep Learning

This file explains how to use the *wtie* package to train and evaluate a deep neural network to estimate a source wavelet given the seismic and reflectivity series.


### Run experiments
To start an experiment from the root of the **well-tie** folder, type the following command in a console:
```shell
./train_network.sh my_experiment
```
The *train_network.sh* script takes a single argument which will be the name given to the created folder. The script will read the parameters in the file *experiments/parameters.yaml* and launch the *experiments/main.py* file. Results will be saved in the folder **experiments/results/my_experiment**.

### Hyper-parameters optimization
To start an hyper-parameter search, type the following command in a console:
```shell
./network_training_hyper_search.sh my_search
```
The *network_training_hyper_search.sh* script takes a single argument which will be the name given to the created folder. The underlying tool running the optimization is the library [Ax](https://github.com/facebook/Ax). The script will read the parameters from the file *experiments/search_space.yaml* and launch the *experiments/hyper_search.py* file. Results will be saved in the folder **experiments/optim/my_search**.


