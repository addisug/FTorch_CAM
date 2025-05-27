# Integrating ML emulators in the  Community Atmosphere Model (CAM)  using FTorch.

## Overview

This guide outlines the process of integrating Machine Learning (ML) emulators into the Community Atmosphere Model (CAM) using FTorch on NCAR's Derecho High-Performance Computing (HPC) system. FTorch serves as a bridge between PyTorch-based ML models and Fortran-based climate models, enabling the replacement or augmentation of traditional physics parameterizations with ML-based alternatives. 

## System Context

This documentation assumes you have:
- An active account on NCAR's Derecho system
- Basic familiarity with HPC environments and module systems

## What is FTorch?

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
```

### 2. Test Model on Derecho

```bash
# Load required modules
module load conda
# if pytorch isn't installed in the conda enviroment, create the environment and install it
create --name pytorch_env
conda activate pytorch_env
conda install pytorch

# Run the Python script
python constant_model.py
```
The model 'constant_model.pt' will be generated

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


Make sure the Fortran interface 'pytorch_test.F90' in the 'src/physics/cam'. 
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

### 5. Compile and RUN

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


###  Example 2

This example provides a simple but complete demonstration of how to use the FTorch library.

###  1. Create the PyTorch Model

The model defines a very simple PyTorch 'net' that takes an input vector of length 5 and applies a single Linear layer to multiply it by 2.

<img src="images/simplenet.png" width="300"/>

Create a file named simplenet.py in your working directory. 
```bash
vi  simplenet.py 
```
The following script consists of a single Linear layer with weights predefined to
multiply the input by 2.

```bash
import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False))
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._fwd_seq(batch)

if __name__ == "__main__":
    model = SimpleNet()
    model.eval()

    # Export to TorchScript format (required for FTorch)
    model_scripted = torch.jit.script(model)

    # Save the model
    model_scripted.save("simplenet_model.pt")

```

### 2. Test Model on Derecho

```bash
# Load required modules
module load conda
# if pytorch isn't installed in the conda enviroment, create the environment and install it
create --name pytorch_env
conda activate pytorch_env
conda install pytorch

# Run the Python script
python simplenet.py
```
The model 'simplenet_model.pt' will be generated


### 3. Create Fortran Interface

CAM-CESM code for this exercise can be obtained as following:
```bash
git clone https://github.com/jedwards4b/cesm.git cesm2.1-alphabranch-ftorch
cd cesm2.1-alphabranch-ftorch/
./manage_externals/checkout_externals
```

Create a file named cam_nn.F90 in your CAM source directory. The recommended location is:
```bash
vi  components/cam/src/physics/cam/cam_nn.F90
```

```bash
module cam_nn
#ifdef USE_FTORCH
  use ftorch,         only: torch_kCPU, torch_tensor, torch_model, torch_tensor_from_array
  use ftorch,         only: torch_model_load, torch_model_forward, torch_tensor_delete
  use camsrfexch,     only: cam_in_t
  use physics_types,  only: physics_state
  use shr_kind_mod,   only: CL=>shr_kind_cl
  !use spmd_utils,     only: masterproc
  use cam_abortutils,only: endrun
  use cam_logfile ,   only:iulog
  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32
  use namelist_utils,  only: find_group_name
  use spmd_utils,     only: masterproc
  implicit none

  public :: torch_inference, torch_readnl

  ! Declare the torch model without initializing it here
  type(torch_model)  :: model
  logical            :: model_initialized = .false.
  character(len=CL) :: weights_file

contains

  subroutine torch_readnl(nlfile)
    use namelist_utils, only : find_group_name
    use units,          only : getunit, freeunit
    use spmd_utils,      only: mpicom, mstrid=>masterprocid, mpi_character, masterproc
    character(len=*), intent(in) :: nlfile
    integer :: unitn
    integer :: ierr
    character(len=*), parameter :: sub="torch_readnl"

    namelist /torch_nl/ weights_file

    if(masterproc) then
       unitn = getunit()
       open( newunit=unitn, file=trim(nlfile), status='old' )
       call find_group_name(unitn, 'torch_nl', status=ierr)
       if (ierr == 0) then
          read(unitn, torch_nl, iostat=ierr)
          if (ierr /= 0) then
             call endrun(sub//': FATAL: reading namelist')
          end if
       end if
       close(unitn)
       call freeunit(unitn)
    endif

    ! Broadcast namelist variables
    call mpi_bcast(weights_file, len(weights_file), mpi_character, mstrid, mpicom, ierr)

  end subroutine torch_readnl

  subroutine init_torch_model(model)
    ! Initialize the model
    type(torch_model), intent(inout) :: model
    character(len=*), parameter :: sub="init_torch_model"

    call torch_model_load(model, weights_file, torch_kCPU)
  end subroutine init_torch_model

  subroutine torch_inference
    implicit none

    ! Set working precision for reals
    integer, parameter :: wp = sp

    type(torch_model) :: model
    type(torch_tensor), dimension(1) :: in_tensors
    type(torch_tensor), dimension(1) :: out_tensors

    real(wp), dimension(5), target :: in_data
    real(wp), dimension(5), target :: out_data

    ! Define tensor shape
    integer      :: tensor_layout(1) = [1]

    ! Initialize the model if it has not been initialized yet
    if (.not. model_initialized) then
       call init_torch_model(model)
       model_initialized = .true.
    end if

    ! Initialise data
    in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

    call torch_model_load(model, trim(weights_file))

    ! Make Torch Tensors for input and output
    call torch_tensor_from_array(in_tensors(1), in_data,    tensor_layout, torch_kCPU)

    call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

    ! Perform inference
    call torch_model_forward(model, in_tensors, out_tensors)

    ! Free the torch tensors
    call torch_tensor_delete(in_tensors(1))

    call torch_tensor_delete(out_tensors(1)) 
  end subroutine torch_inference
#endif
end module cam_nn

```

In your Fortran interface file, the path to the PyTorch model is indicated through the namelist, and you need to edit the following files:

```bash
vi  components/cam/bld/namelist_files/namelist_definition.xml
```
Append the following text at the end of the file

```bash
<entry id="weights_file" type="char*256" input_pathname="abs" group="torch_nl" valid_values="">
  Path to Neural Network weights file
        Default: set by build-namelist
</entry>
```
```bash
vi components/cam/src/control/runtime_opts.F90
```

Add the following line the few lines of the script
```bash 
use cam_nn,              only: torch_readnl
```

### 4. Integrate with CAM

Make sure the Fortran interface 'cam_nn.F90' in the 'src/physics/cam'. 
Then edit 'physpkg.F90' file:

```bash
vi src/physics/cam/physpkg.F90
```
Insert the following line at the beginning of the file
```fortran
#ifdef USE_FTORCH
  use cam_nn,             only: torch_inference
#endif
```

At the end of subroutine 'phys_init', add the following:
```fortran
 ! Test for neural network initialization
 #ifdef USE_FTORCH
    call torch_inference 
 #endif
```

## 5. Compile and RUN

```bash

```
Compile the CAM model, in this case we used gnu compiler. 
```bash
./cime/scripts/create_newcase --case /path/test_ftorch --mach derecho --compiler gnu --compset FHIST --res f09_f09_mg17 --project your project number
```

You need to specify the path of pytorch model in your 'user_nl_cam' file in the case directory, you need to write:

```bash
 vi user_nl_cam
 weights_file = '/path/simplenet_model.pt'
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
