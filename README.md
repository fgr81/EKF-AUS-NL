## DISCLAIMER

This is a sperimental branch, it is not working.
Pybind11 allow using EKF_AUS instances inside python scripts, as slam.py .


## Synopsis

This library implements the EKF-AUS-NL ( Extended Kalman Filter with Assimilation confined in the Unstable Space) algorithm, presented by A. Trevisan and L. Palatella in

Trevisan, A., and L. Palatella. "On the Kalman Filter error covariance collapse into the unstable subspace." Nonlinear Processes in Geophysics 18.2 (2011): 243-250.

and

Palatella, Luigi, and Anna Trevisan. "Interaction of Lyapunov vectors in the formulation of the nonlinear extension of the Kalman filter." Physical Review E 91.4 (2015): 042905.

## Motivation

In this project we propose a variation of the algorithm EKF-AUS-NL designed to perform the data assimilation process when the Jacobian ∂F_i(x)/∂x_j can not be calculated and/or defined. Moreover we propose a simple approach to be followed in order to take into account the presence of parametric model error in the framework of the EKF-AUS-NL routines.

## Installation

Wanting to use Conda:
```
conda env create -f ekf.yml
conda activate ekf
```

Then type:
```
cmake .
make
```

Finally:
```
python slam.py
```








This implementation of EKF-AUS-NL could be applied to several systems, described by proper dynamical equations. In ./external there are the implementations of two systems: L96 and SLAM; they can be used to test the filter. The main routine manages these two options on command line; the user has to indicate also a text file that include the initial condition, e.g. initial_SLAM.dat and initial_l96.dat.

Compilation

./make

Different run options:

to obtain a brief help:

./EKF-AUS-NL

full help

./EKF-AUS-NL -h

To tun on L96 model:

./EKF-AUS-NL L96 initial_l96.dat > log &

To tun on SLAM:

./EKF-AUS-NL SLAM initial_SLAM.dat > log &


The log's of the assimilation algorithm can be found in the files log and AssimilationLog.dat .


## API Reference

This project is documented by doxygen, you could find the documentation in the directory ./doxygen .
The algorithm is implemented in the class EKF_AUS, you could see how you can use it by looking the implementations of the interface IAssimilate: SLAM_Assimilated and L96_Assimilated.
If the user wants to apply the filter to another dynamical system,
he/she has to implement a custom class containing the definition of all the proper methods declared in the interface class IAssimilate. These methods regard the dynamical details of the model that the user want to use in the assimilation test. These routines are: evolve, readFile, writeFile, dimensions (giving the dimensionality N of the state vector starting from the state file of the model) .
These diagrams show the interaction between these routines:

![Sequence diagram #1](sequence-dia-1.png "Sequence Diagram #1")

![Sequence diagram #2](sequence-dia-2.png "Sequence Diagram #2")


## Contributors

Luigi Palatella, Fabio Grasso. For algorithm details pleas ask to luigi.palatella@yahoo.it

## License

Mozilla Public License 2.0 (MPL-2.0)
