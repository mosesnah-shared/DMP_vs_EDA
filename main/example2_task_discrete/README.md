## Goal-directed Discrete Movement in Task-Space
This Github directory contains the MuJoCo-python codes for the simulation in Section 3.3 of the [**Robot Control based on Motor Primitives-A Comparison of Two Approaches**](https://arxiv.org/abs/2310.18771) written by [Moses C. Nah](https://mosesnah-shared.github.io/about.html), [Johannes Lachner](https://jlachner.github.io/) and [Neville Hogan](https://meche.mit.edu/people/faculty/neville@mit.edu). 
This code is to generate goal-directed discrete movement in task-space, without kinematic redundancy.

 To run DMP:
```bash
    python -B DMP.py
```
To run EDA:
```bash
    python -B EDA.py
```

Example results are shown below.
![Example1_joint_discrete](../../MATLAB/gifs/example2a.gif)
For DMP, perfect tracking in task-space is achieved, whereas for EDA, a non-negligible displacement exists. 

As pointed out in the [paper](https://arxiv.org/abs/2310.18771), one can reduce the tracking error for EDA, by increasing the task-space stiffness and damping matrices.  
An example is shown here, where if one uses a higher task-space stiffness, the tracking error reduces. 
![Example1_joint_discrete](../../MATLAB/gifs/example2c.gif)

## Managing Kinematic Singularity
While EDA has no problem with numerical stability near (or even at) kinematic singularity, DMP needs an additional method to handle kinematic singularity. This is due to the fact that DMP requires to solve the Inverse Kinematics, whereas EDA does not.
An example application is shown below. For DMP, Damped Least-square inverse is used to maintain stability near (or at) kinematic singularity, whereas for EDA the robot even passed through kinematic singularity and achieve the goal-directed discrete movement.

![Example1_joint_discrete](../../MATLAB/gifs/example2b.gif)

