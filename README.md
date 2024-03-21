# Dynamic Movement Prmitives (DMP) vs. Elementary Dynamic Actions (EDA)

This Github repository contains the MuJoCo-python codes for running the simulations presented in the Manuscript: [**Robot Control based on Motor Primitives-A Comparison of Two Approaches**](https://arxiv.org/abs/2310.18771) written by [Moses C. Nah](https://mosesnah-shared.github.io/about.html), [Johannes Lachner](https://jlachner.github.io/) and [Neville Hogan](https://meche.mit.edu/people/faculty/neville@mit.edu). 

We assume the usage of `venv` for the code, and also Python3. If Python3 is not installed, please download and install it from [python.org](https://www.python.org/).

## Descriptions
The list of simulations are as follows:
- [Example 1: Discrete movement in Joint-space](./main/example1_joint_discrete)
- [Example 2: Discrete movement in Task-space, Position](./main/example2_task_discrete)
- [Example 3: Managing Unexpected Physical Contact](./main/example3_unexpected_contact)
- [Example 4: Obstacle Avoidance](./main/example4_obstacle_avoidance)
- [Example 5: Rhythmic Movement, both in Joint-space and Task-space](./main/example5_rhythmic)
- [Example 6: Combination of Discrete and Rhythmic Movements, both in Joint-space and Task-space](./main/example6_discrete_and_rhythmic)
- [Example 7: Sequencing Discrete Movements](./main/example7_sequencing)
- [Example 8: Managing Kinematric Redundancy](./main/example8_redundancy)
- [Example 9: Discrete movement in Task-space, Both Position and Orientation](./main/example9_pos_and_orient)

All folders include codes for both Dynamic Movement Primitives (DMP) and Elementary Dynamic Actions (EDA), and each code is heavily commented for reproducability.

## Getting Started
To run the code, please follow the instructions below.

### Setting Up a Virtual Environment

It's recommended to create a virtual environment for the project dependencies. To set up a virtual environment named `.venv`, follow these steps:

1. Navigate to your project directory in the terminal.

2. Run the following command to create a virtual environment:

```bash
python -m venv .venv
```

This command creates a `.venv` directory in your project directory, which will contain the Python executable files and a copy of the pip library.

### Activating the Virtual Environment

Before you can start installing or using packages in your virtual environment, you need to activate it. Follow the instructions for your operating system:

#### On Windows

```bash
.venv\Scripts\activate
```

#### On macOS and Linux

```bash
source .venv/bin/activate
```

This will return your terminal to its normal state.

## Installing Dependencies
With your virtual environment activated, install project dependencies by running:
```bash
pip install -r requirements.txt
```
Note that all the packages required to run the code are already in the `requirements.txt`. 
Moreover, please make sure that your `pip` is the recent version by typing:
```bash
pip install --upgrade pip
```


## Running the Application
Once the installation is complete, please do check whether the following code is executable.
```bash
python ./main/example1_joint_discrete/EDA.py
```
If it is successful, you are all set!