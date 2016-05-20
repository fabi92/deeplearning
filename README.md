WELCOME TO MY DEEP LEARNING GIT REPOSITORY
----------------------------------------------------------------------

* Introduction
* Implementations
* Requirements

INTRODUCTION
------------

This Repository includes all files for the provided assignments.
For each task a respective folder will be submitted. Plus, scripts which are used within all assignemnts. Please note that sanity checks are neglected.

IMPLEMENTATIONS
---------------

In the results folder of each implementation, one can find all plots regarding error, receptive fields and so on.
All the invokable scripts are explained the documentation. There you can find descriptions and all parameters the scripts require. However, the argparse module explains everything as well if you call the script with help (as for instance trainAutoEncoder.py --help).

* nn<br />
  Contains The Feed-Forward Neural Network implementation<br />
  Although, only 2 layers are requested (and also standard), more layers can be passed to the script via the comand line interface.
* logreg<br />
  Contains the implementation for the Logistic Regression task, plus a second python and shell script which holds the same implemenation using climin and in the comand line provided optimization technic.
* latent
  - Autoencoder<br />
  Contains the python scripts plus the reconstrcution and receptive fields refering to the respective regularization.
  - PCA<br />
  Contains the python scripts and the created scatter plots for mnist and cifar.
* kmeans<br />
  Each Folder Contains two files. The kmeans classes and the scripts to run the implementations on a set. Further, in the results folder all figures which were created can be found.<br />
  - Coates kMeans
  - Minibatch kMeans
  - Kulis kMeans
* Tools<br />
  Contains all scripts which are required by multiple solutions. These folder needs to be added to the PYTHONPATH variable.
  - Plotter<br />
      Creates all plots using matplotlib
  - LoadData<br />
      Contains all routines for loading the Mnist or Cifar data set. Furthermore, it provides more functions for each with respect to color or set splits.
  - Cost<br />
      Contains all cost functions which are used.

REQUIREMENTS
------------

In order to be able to safely run any submitted script, note that it usually
is necessary to source the environments script, first. It can be found in the
base directory of this repository.
Further, to source it, simply run  *source environment.sh*
Basically, the environmenct script adds the tools folder to the PYTHONPATH variable. However, this can also be done manually.<br />
Furthermore, you are expected to pass the dataset in the following way:
* MNIST
  - Forward path to the pkg file to the script
* CIFAR10
  - Forward path to the folder containing the python pickled batches
