# ML Emulator Integration with CAM using FTorch

## Overview
This guide provides instructions for integrating Machine Learning (ML) emulators using FTorch within the Community Atmosphere Model (CAM). FTorch enables seamless integration of PyTorch-based ML models into Fortran-based climate models.

## Prerequisites
- CAM model (tested with version X.X)
- FTorch library
- PyTorch (version X.X)
- Python 3.x
- Fortran compiler (gfortran/ifort)
- MPI library
- NetCDF library

## Installation

### 1. FTorch Setup
```bash
git clone https://github.com/your-repo/ftorch
cd ftorch
make install
```

### 2. Configure CAM Build
Add the following to your CAM configuration:
```bash
./configure --with-ftorch=/path/to/ftorch
```

## Integration Steps

### 1. Prepare ML Model
1. Train your PyTorch model
2. Export to TorchScript format:
```python
import torch

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model_weights.pth'))

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')
```

### 2. Modify CAM Source
Add the following to your CAM physics module:
```fortran
! In your physics module
use ftorch_module, only: ftorch_model

type(ftorch_model) :: ml_emulator

! Initialize model
call ml_emulator%load('path/to/model.pt')

! Use model in physics calculations
subroutine compute_physics(state)
    real, intent(inout) :: state(:,:)
    real :: ml_output(size(state,1))
    
    ! Call ML model
    call ml_emulator%forward(state, ml_output)
    
    ! Use ML output in physics
    state = ml_output
end subroutine
```

### 3. Configure Runtime Settings
Add to your namelist:
```
&cam_inparm
 use_ml_emulator = .true.
 ml_model_path = 'path/to/model.pt'
/
```

## Usage

### Running with ML Emulator
1. Build CAM with ML support:
```bash
./case.build
```

2. Submit job:
```bash
./case.submit
```

### Verification
Check model output for ML integration:
```bash
tail -f cam.log.*
```

Look for:
```
ML Emulator initialized successfully
```

## Debugging

### Common Issues
1. Model Loading Errors
   - Verify model path in namelist
   - Check TorchScript compatibility
   - Validate FTorch installation

2. Runtime Errors
   - Check input/output tensor dimensions
   - Verify memory allocation
   - Monitor MPI communication

### Debug Mode
Enable debug output:
```bash
./xmlchange CAM_DEBUG=TRUE
./xmlchange DEBUG=TRUE
```

## Performance Considerations
- ML model inference adds computational overhead
- Consider batch processing for efficiency
- Monitor memory usage with large models
- Test scaling behavior with different MPI configurations

## Contributing
Please submit bug reports, feature requests, and pull requests through GitHub.

## License
[Your License]

## Contact
[Your Contact Information]

## Acknowledgments
- FTorch development team
- CAM development team
- [Other acknowledgments]
