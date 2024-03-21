# Dynamic Movement Prmitives (DMP) vs. Elementary Dynamic Actions (EDA)
This Github repository contains the MuJoCo-python codes for the simulation in Section 3.2 of the [**Robot Control based on Motor Primitives-A Comparison of Two Approaches**](https://arxiv.org/abs/2310.18771) written by [Moses C. Nah](https://mosesnah-shared.github.io/about.html), [Johannes Lachner](https://jlachner.github.io/) and [Neville Hogan](https://meche.mit.edu/people/faculty/neville@mit.edu). 

![Example1_joint_discrete](../MATLAB/gifs/example1.gif)


# Dynamic Movement Prmitives (DMP)
For Dynamic Movement Primitives, DMP for discrete movement is used. 
Using Imitation Learning, the desired joint trajectory for position $\mathbf{q}_{des}(t)$, velocity $\dot{\mathbf{q}}_{des}(t)$ and acceleration $\ddot{\mathbf{q}}_{des}(t)$ are derived. 
Once these terms are derived, the torque input is given from the Inverse Dynamics Model:
$$
    \bm{\tau}(t) = \mathbf{M}(\mathbf{q}_{des}(t))\ddot{\mathbf{q}}_{des}(t) + \mathbf{C}(\mathbf{q}_{des}(t), \dot{\mathbf{q}}_{des}(t))\dot{\mathbf{q}}_{des}(t) 
$$

# Elementary Dynamic Actions (EDA)
The input torque command to the robot is simply a first-order joint-space impedance controller (or joint PD controller, given an ideal torque-source actuator):
$$
    \bm{\tau}(t) = \mathbf{K}_q (\mathbf{q}_0(t)-\mathbf{q}(t)) + \mathbf{B}_q(\dot{\mathbf{q}}_0(t) - \dot{\mathbf{q}}(t))
$$
The transparent robot shown in Elementary Dynamic Actions (right-side) is the Virtual robot configuration defined by $\mathbf{q}_0(t)$. 
