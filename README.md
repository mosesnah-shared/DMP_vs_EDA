# Dynamic Movement Prmitives (DMP) vs. Elementary Dynamic Actions (EDA)

This Github repository contains the MuJoCo-python codes for running the simulations presented in this paper.
We assume the usage of `venv` for the code, and also Python3.
If Python3 is not installed, please download and install it from [python.org](https://www.python.org/).

## Details
The code contains the details for 

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


## Running the Application
Once the installation is complete, please do check whether the following code is executable.
```bash
pip install -r requirements.txt
```
If the code successfully runs, then you are all set! Please do try out all the codes 
