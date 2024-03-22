## Goal-directed Discrete Movement in Joint-Space
This Github repository contains the MuJoCo-python codes for the simulation in Section 3.2 of the [**Robot Control based on Motor Primitives-A Comparison of Two Approaches**](https://arxiv.org/abs/2310.18771) written by [Moses C. Nah](https://mosesnah-shared.github.io/about.html), [Johannes Lachner](https://jlachner.github.io/) and [Neville Hogan](https://meche.mit.edu/people/faculty/neville@mit.edu). 
This code is to generate Goal-directed discrete movement in joint-space.


To run the example for DMP:
```bash
    python -B DMP.py
```
To run the example for EDA:
```bash
    python -B EDA.py
```

An example result for DMP and EDA are shown below. For DMP, perfect tracking in joint-space is achieved, whereas for EDA, a non-negligible displacement exists.
![Example1_joint_discrete](../../MATLAB/gifs/example1a.gif)


As pointed out in the [paper](https://arxiv.org/abs/2310.18771), one can reduce the tracking error for EDA, by increasing the joint-space stiffness and damping matrices.  
An example is shown here, where if one uses a higher joint-space stiffness, the tracking error reduces. 

![Example1_joint_discrete](../../MATLAB/gifs/example1b.gif)