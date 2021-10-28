# Automatic Seismic to Well Tie

Author: *Valentin Tschannen* - Fraunhofer ITWM, Germany.

This package aims to provide basic utilities to automate the seismic to well tie process.
Part of the package is dedicated to the creation and training of a neural network to perform wavelet extraction. It also contains utilities to create synthetic training data.
The other modules are dedicated to the remaining operations (log processing, depth to time conversion), and use random search and bayesian optimization for the automatic tuning of hyper-parameters.

### Package content
* README.md, Makefile, setup.py, environment.yml, main.sh
* **scripts** : shell scripts for the installation
* **tests** : unit/intergration tests
* **documentation** : documentation built during installation
* **notebooks** : jupyter notebooks demonstrating the use of the library
* **experiments** : config files to train a neural network and folder where results are saved
* **wtie** :  python package


### Installation
#### Linux
(TODO: docker)
Python dependencies are listed in the file *enivronment_linux.yml*. Assuming that you installed the [Anaconda platform](https://www.anaconda.com/) and that your system's os is unix-like with the tool [make](https://www.gnu.org/software/make/), you can install the *wtie* package by running the following command in a **shell console** in your *base* environment (alternatively, take a look at the *Makefile* to see the steps that you need to reproduce to install on your system):
```shell
make install
```
This will create a conda environment named *wtie* and install the package (this may take several minutes, especially if you are going through a network adapter).
If there were no problems, you are ready to work with the package!

Don't forget to activate the environment before working with the package:
```shell
conda activate wtie
```

#### Windows
(TODO: better Windows support)
Python dependencies are listed in the file *enivronment_windows.yml*. For this guide I assume that you installed the [Anaconda platform](https://www.anaconda.com/). Go to the main well-tie directory and go through the follwoing steps:

First, open an Anaconda prompt and create a new conda environement:
```shell
conda env create -f environment_windows.yml
```
This will create a conda environment named *wtie*.

Second, open a Windows command prompt and activate the environment:
```shell
conda activate wtie
```

Then, in the same prompt, install the python *wtie* package:
```shell
python setup.py develop
```

Unfortunatly, Windows does not seem to be able to streamline the installation of the [noise](https://github.com/caseman/noise) package that we use to generate correlated synthetic reflectivity series. To install this package you should follow instructions given [here](https://stackoverflow.com/questions/53365282/cant-install-noise-module).

Finally, from the same prompt, move to the **tests** folder and run the test suite:
```shell
pytest -v --basetemp="./tmp"
```

Don't forget to activate the environment everytime you want to work with the package:
```shell
conda activate wtie
```


### Tutorial
A series of [**notebooks**](./notebooks) demonstrates the use of the library to perform an automatic well tie. The weights of a pretrained network as well as data coming from the open [Volve](https://www.equinor.com/en/what-we-do/digitalisation-in-our-dna/volve-field-data-village-download.html) and [Poseidon](https://terranubis.com/datainfo/NW-Shelf-Australia-Poseidon-3D
) datasets are provided in the [data](./data/tutorial) folder.

Explanations about how to train your own neural network are given [here](./wtie/learning/readme.md).

## Publications
A publication titled "Partial Automation of the Seismic to Well Tie with Deep Learning and Bayesian Optimization" is under review. In the mean time, if this package is useful in your work, you may cite the following:
- Valentin Tschannen. Deep Learning for Seismic Data Processing and Interpretation. Ph.D. Thesis. École normale supérieure, Université PSL, 2020. [link](https://hal.archives-ouvertes.fr/tel-02962268)
