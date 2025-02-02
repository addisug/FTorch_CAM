# Integrating ML emulators in the  Community Atmosphere Model (CAM)  using FTorch.

## Overview

This guide outlines the process of integrating Machine Learning (ML) emulators into the Community Atmosphere Model (CAM) using FTorch on NCAR's Derecho High-Performance Computing (HPC) system. FTorch serves as a bridge between PyTorch-based ML models and Fortran-based climate models, enabling the replacement or augmentation of traditional physics parameterizations with ML-based alternatives. 

## System Context

This documentation assumes you have:
- An active account on NCAR's Derecho system
- Basic familiarity with HPC environments and module systems

## What is FTorch

FTorch is a specialized interface library that:

- Enables direct calling of PyTorch models from Fortran code
- Handles data type conversions between PyTorch tensors and Fortran arrays
- Manages computational resource allocation for ML inference
- Execute the model on either the CPU or GPU, ensuring seamless data handling from the respective hardware.
- Prioritize minimal performance overhead for optimal efficiency.

Visit https://github.com/alexeedm/pytorch-fortran to learn more about FTorch

##  Integration Benefits

- Replace computationally expensive physics parameterizations with efficient ML emulators
- Maintain the existing CAM workflow while incorporating ML capabilities
- Leverage PyTorch's extensive ML ecosystem within Fortran-based climate models
- Enable hybrid modeling approaches combining traditional physics with ML

###  Example 1 
It is the most basic type of PyTorch model integration with CAM using FTorch. 
We'll create a constant model that always returns ones, serving as a minimal working example.

###  1. Create the PyTorch Model

```bash
# constant_model.py
import torch
import torch.nn as nn

class ConstantModel(nn.Module):
    def __init__(self):
        super(ConstantModel, self).__init__()
    
    def forward(self, x):
        # Always return a tensor of ones with the same shape as input
        return torch.ones_like(x, dtype=torch.float)

def create_and_save_model():
    # Instantiate the model
    model = ConstantModel()
    
    # Export to TorchScript format (required for FTorch)
    model_scripted = torch.jit.script(model)
    
    # Save the model
    model_scripted.save("constant_model.pt")
    
    # Test the model
    example_input = torch.tensor([0.5])
    output = model(example_input)
    print(f"Test input: {example_input}")
    print(f"Test output: {output}")

if __name__ == "__main__":
```

### 2. Test Model on Derecho

```bash
# Load required modules
module load conda
conda activate pytorch_env

# Run the Python script
python constant_model.py
```

### 3. Create Fortran Interface

CAM-CESM code for this exercise can be obtained as following:
```bash
git clone https://github.com/jedwards4b/cesm.git cesm2.1-alphabranch-ftorch
cd cesm2.1-alphabranch-ftorch/
./manage_externals/checkout_externals
```

Create a file named pytorch_test.F90 in your CAM source directory. The recommended location is:
```bash
vi  components/cam/src/physics/cam/pytorch_test.F90
```

In your Fortran interface file, specify the path to your PyTorch model:

```fortran
module pytorch_test

  use shr_kind_mod, only: r8 => shr_kind_r8

  use ftorch, only : torch_model, torch_model_load, torch_model_forward, &
          torch_tensor, torch_tensor_from_array, torch_kCPU,  torch_delete

  use iso_fortran_env

  implicit none
  save

  character(len=256) :: cb_torch_model = "/path/constant_model.pt"
  type(torch_model) :: model_pytorch

  public init_neural_net, neural_net

contains

  subroutine init_neural_net()
    implicit none
    call torch_model_load(model_pytorch, trim(cb_torch_model))
  end subroutine init_neural_net

  subroutine neural_net()
     implicit none
     integer              :: in_layout(1) = [1]
     integer              :: out_layout(1) = [1]

     ! NN variables
     type(torch_tensor), dimension(1) :: in_tensor, out_tensor
     real(real32) :: in_data(1) = 1.0
     real(real32), dimension(1) :: out_data

     call torch_tensor_from_array(in_tensor(1), in_data, in_layout, torch_kCPU) ! Ftorch
     call torch_tensor_from_array(out_tensor(1), out_data, out_layout, torch_kCPU)
    call torch_model_forward(model_pytorch, in_tensor, out_tensor)

  end subroutine neural_net

end module pytorch_test

```

### 4. Integrate with CAM

To obtain the CAM_CESM code, you need to do the following:  
```bash
git clone https://github.com/jedwards4b/cesm.git cesm2.1-alphabranch-ftorch
cd cesm2.1-alphabranch-ftorch/
./manage_externals/checkout_externals
```

Add the Fortran interface 'pytorch_test.F90' in the 'src/physics/cam'. 
Then edit 'physpkg.F90' file:

```bash
vi src/physics/cam/physpkg.F90
```
Insert the following line at the beginning of the file
```fortran
use pytorch_test,        only: init_neural_net, neural_net
```

At the end of subroutine 'phys_init', add the following:
```fortran
 ! Test for neural network initialization
    call init_neural_net()

    ! Test for neural network run
    call neural_net()
```

### 6. Compile and RUN

Compile the CAM model, in this case we used gnu compiler. 
```bash
./cime/scripts/create_newcase --case /path/test_ftorch --mach derecho --compiler gnu --compset FHIST --res f09_f09_mg17 --project your project number
```


Export FTorch environment:
```bash
export USE_FTORCH=TRUE
export FTORCH_PREFIX_FTORCH=/root/ftorch/install/path
export CONDA_PREFIX=FALSE
```
Setup, build and submit

```bash
./case.setup
./case.build 
./xmlchange STOP_OPTION=ndays
./xmlchange STOP_N=1
./xmlchange RESUBMIT=0
./xmlchange JOB_WALLCLOCK_TIME=00:20:00
./xmlchange PROJECT=your project number
./case.submit
```