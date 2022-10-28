# Nektar simulations

This folder contains files for Nektar++ simulations.

Flow case:

- unsteady 2D cylinder flow at Re=150.

This folder contains:

- 6 folders with convergence cases with Nektar++ simulation definition file and mesh definition file, as well as output file with forces vs iteration,
- convergence plot folder that contains the code for plotting the convergence plots from the forces data,
- a folder with a rerun case to gather the mean flow data,
- data_analysis_transformation folder with code for output data analysis and transformation.


Additionally, cyl.stp file contains cylinder geometry definition used by Nekar++ mesher.

## Simulation procedure

1. Nektar++ simulations were performed.
2. Convergence plot was created using plot.py script in convergence_plots folder.
3. cycle_analysis.py script was used to find cycle length expressed in iterations.
4. Additional rerun was performed with mean data filter to extract mean u,v,p,uu,uv,vv fields.

All transformations below are done on time-averaged fields.

5. The solution file was separated into individual fields for velocity, pressure and reynolds stresses using FieldConvert removefield module as following.
   
        # Extracting velocity fields
        FieldConvert -m removefield:fieldname="p,uu,uv,vv" cyl.xml AvergageField.fld velocities.fld
        # Extracting pressure field
        FieldConvert -m removefield:fieldname="u,v,uu,uv,vv" cyl.xml AvergageField.fld pressure.fld
        # Extracting Reynolds stresses fields
        FieldConvert -m removefield:fieldname="u,v,p" cyl.xml AvergageField.fld stresses.fld

6. Gradients of the fields were computed (output files contain both source field and gradient fields) using FieldConvert gradient module. We need second order gradients for velocities and stresses and first order gradient for pressure.

        FieldConvert -m gradient cyl.xml velocities.fld velocities-1st-grad.fld
        FieldConvert -m gradient cyl.xml velocities-1st-grad.fld velocities-2nd-grad.fld
        FieldConvert -m gradient cyl.xml pressure.fld pressure-1st-grad.fld
        FieldConvert -m gradient cyl.xml stresses.fld stresses-1st-grad.fld
        FieldConvert -m gradient cyl.xml stresses-1st-grad.fld stresses-2nd-grad.fld

7. To manage filesize, we remove base fields and first order derivatives of the Reynolds stresses from the stresses-2nd-grad.fld. That way, stresses-1st-grad.fld contains uu,uv,vv and the first order gradients and stresses-1st-grad.fld contains the second order gradients.

        FieldConvert -m removefield:fieldname="uu,uv,vv,uu_x,uu_y,uv_x,uv_y,vv_x,vv_y" cyl.xml stresses-2nd-grad.fld


8. Shear stresses were computed by extracting mesh of the cylinder surface and using FieldConvert wss module.

        # Extracting shear stresses on boundary 0 (cylinder)
        FieldConvert -m wss:bnd=0:addnormals=0 cyl.xml AvergageField.fld cyl_wss.fld 
        # Extracting mesh of the boundary
        NekMesh -m extract:surf=5,6,7,8 cyl.xml bl.xml
        # Extracting point data
        FieldConvert bl.xml cyl_wss.fld cyl_wss.dat

    cyl_wss.dat is a file that contains shear stress distribution over the cylinder surface.

9.  Pressure distribution over the cylinder surface is done similarily but this time using extract module.

        # Extracting pressure data on boundary 0 (cylinder)
        FieldConvert -m extract:bnd=0 cyl.xml AvergageField.fld pressure-boundary.fld 
        # Extracting mesh of the boundary (already done)
        NekMesh -m extract:surf=5,6,7,8 cyl.xml bl.xml
        # Extracting point data
        FieldConvert bl.xml pressure-boundary.fld pressure-boundary.dat

    pressure-boundary.dat is a file that contains static pressure distribution over the cylinder surface.

10. point_gen.py script was used to define points for field interpolation - file with output points is named 'cyl.pts'.
11. The field data files were interpolated for the given points using FieldConvert interppoints module.
   
        # Velocities
        FieldConvert -m interppoints:fromxml=cyl.xml:fromfld=velocities-2nd-grad.fld:topts=cyl.pts velocities.dat
        # Pressure
        FieldConvert -m interppoints:fromxml=cyl.xml:fromfld=velocities-2nd-grad.fld:topts=cyl.pts pressure.dat
        # Stresses
        FieldConvert -m interppoints:fromxml=cyl.xml:fromfld=velocities-1st-grad.fld:topts=cyl.pts stresses.dat
        FieldConvert -m interppoints:fromxml=cyl.xml:fromfld=velocities-2nd-grad.fld:topts=cyl.pts stresses_2nd.dat

#### Summary of files for PINN training and error analysis:

- velocities.dat - contains time-avergaged velocity fields and their 1st and 2nd order gradients
- pressure.dat - contains time-avergaged pressure field and its gradient
- stresses.dat - contains time-avergaged Reynolds stresses fields and their 1st order gradients
- stresses.dat - contains 2st order gradients of time-avergaged Reynolds stresses fields
- cyl_wss.dat - contains time-avergaged shear stresses distribution over the cylinder (for error analysis only)
- pressure-boundary.dat - contains time-avergaged pressure distribution over the cylinder (for error analysis only)
