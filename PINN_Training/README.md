# PINN Training

This folder contains PINN training code.

## Description of case_\*.py files

Each case_\*.py corresponds to a single numerical experiment with PINNs for case of unsteady cylinder at Reynolds number 150. The data used comes from [Nektar simulations](../../NektarSimulations/) described in another folder in this repository.

### Structure of case_*.py file

The structure of each case file is very similar. After some helper functions, there is a main() function which contains the most important portion of the numerical experiment. As the script uses DeepXDE library, the numerical expoeriment is analogical to the examples from the DeepXDE repository, available at https://github.com/lululxvi/deepxde.

The PINN experiment procedure is as follows:

1. Name the case and prepare directory for the results.
2. Extract the appriopriate data using the helper function.
3. Define the geometry of the domain.
4. Obtain the PIV points in the specified domain.
5. Define boundary conditions specific to the case.
6. Define collocation points in the domain for the physics loss.
7. Define the governing equation and define an object which binds the domain to the boundary conditions and the governing equations.
8. Create the neural network that maps point in the domain to the field variables.
9. Create the model which binds the neural network mapping with the previously defined object that holds the domain and physics information.
10. Perform NN training.
11. Test the obtained neural network mapping.

The scripts include an option to only run training or testing should that be necessary. By default, both actions are performed - neural network is trained and then the obtained mapping is tested.

### Unsteady cylinder cases description.

- case_unCyl_piv_15 
    - This case corresponds to regression using the forcing formulation of the RANS equation. 
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - PIV data: PIV data contains mean velocities with resolution 0.02x0.02.
- case_unCyl_piv_21 
    - This case corresponds to regression using the forcing formulation of the RANS equation. 
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface. Additionally, the forcing and the curl of forcing is zero at the domain inlet (steady inlet assumption).
    - PIV data: PIV data contains mean velocities with resolution 0.02x0.02.
- case_unCyl_piv_46 
    - This case corresponds to superresolution using the forcing formulation of the RANS equation and the PIV data with mean velocities.
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - PIV data: PIV data contains mean velocities with resolution 0.5x0.5.
- case_unCyl_piv_51 
    - This case corresponds to regression using the expanded formulation of the RANS equation that includes the Reynolds stresses and with PIV data containing the 2nd order statistics of the velocity i.e. Reynolds stresses.
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - PIV data: PIV data contains mean velocities and Reynolds stresses with resolution 0.5x0.5.
- case_unCyl_piv_47 
    - This case corresponds to superresolution using the expanded formulation of the RANS equation that includes the Reynolds stresses, PIV data containing the 2nd order statistics of the velocity i.e. Reynolds stresses and the pressure data over the cylinder surface.
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - PIV data: PIV data contains mean velocities and Reeynolds stresses with resolution 0.5x0.5.
    - Pressure data: pressure values over the cylinder surface.

- cases 46a to 46i
    -  This case corresponds to superresolution using the forcing formulation of the RANS equation and the PIV data with mean velocities.
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - Varying PIV data resolution:
      - 46a - 0.02x0.02
      - 46b - 0.05x0.05
      - 46c - 0.1x0.1
      - 46d - 0.25x0.25
      - 46e - 0.5x0.5
      - 46f - 0.75x0.75
      - 46g - 1.0x1.0
      - 46h - 0.6x0.6
      - 46i - 0.7x0.7
- cases 51a to 51h
    - The same case as 51 but with varying resolution of the PIV data:
      - 51a - 0.02x0.02
      - 51b - 0.05x0.05
      - 51c - 0.1x0.1
      - 51d - 0.25x0.25
      - 51e - 0.5x0.5
      - 51f - 0.6x0.6
      - 51g - 0.7x0.6
      - 51h - 1.0x1.0
  - cases 52-54
    - This case correspond to regression using the forcing formulation on PIV data containing 1st order velocity statistics (mean u and v velocity components) with added noise.
    - Boundary conditions: forcing and velocities are 0 at the cylinder surface.
    - PIV data: PIV data contains mean velocities with resolution 0.02x0.02.
    - Noise: Gaussian at each measurement with specified variance.
    - For:
      - case 52, the variance (level) of noise is 0.02
      - case 53, the variance (level) of noise is 0.05
      - case 54, the vairance (level) of noise is 0.10


### Other files

- format_data.py contains code to extract higher order field quantities from Nektar++ output,
- utilities.py contains helper functions for running PINN regressions,
- equations.py contains deinitions of the governing equations in the form expected by DeepXDE,
- plot_true.py is used to plot the true fields based on Nektar++ simulations data.


## Pre-requisites

Firstly, one needs to obtain data by simulating the flow using Nektar++ and postprocessing the output. See Nektar_Simulations folder in the root directory.
The output files from Nektar++ are:

- velocities.dat - contains time-avergaged velocity fields and their 1st and 2nd order gradients
- pressure.dat - contains time-avergaged pressure field and its gradient
- stresses.dat - contains time-avergaged Reynolds stresses fields and their 1st and 2nd order gradients
- cyl_wss.dat - contains time-avergaged shear stresses distribution over the cylinder (for error analysis only)
- pressure-boundary.dat - contains time-avergaged pressure distribution over the cylinder (for error analysis only)

All of the files have to be place inside 'Data' folder and 'format_data.py' script has to be run to transform .dat files into .mat files used in PINN training.
