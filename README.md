# Massively Parallel GPU-accelerated String Method
This reposiotry contains the executable file for massively paralle GPU-accelearated string method to obtain the minimum energy path for rigid molecule in nanoporous materials.

&copy;All rights reserved


## Current Status
We are still actively developing this program and extending it to more systems (e.g. considering the flexbile gas molecules and framework materials). Any collaboration or interest is welcome. For any interest, please contact Musen Zhou (mzhou035@ucr.edu) and Jianzhong Wu (jwu@engr.ucr.edu).

## Requirement
In order to use this executable file, Nvidia CUDA toolkit is required. Check CUDA availability and version 'nvcc --version'. The current version has been tested under CUA V11.1.105.

## Install
After cloning this repository, the program would be installed simple by executing 'install.bash'.

## Run the example
After uploading all the files, run the calculation simply by the following command:

'./GPU_string_polyatmoic example.input output.dat'

## Input File
The file 'example.input' is an example input file. It demonstrates the input of minimum energy path calculation for ethene in MOF-5.


## Output File
Currently, the output file lists the fractional coordinate and euler angles at the center of mass for the molecule and correspoding external potential, which allows the calculation of self-diffusivity coefficient at infinite dilution.

## Convergence Criteria
In this current release, convergence criteria is set as |r^{t}-r^{t+1}|<5e-5 and |$\omega$^{t}-$\omega$^{t+1}|<0.5 for the given sample input.

## Diffusivity Calculation
diffusivity.c can be used to calculate the diffusivity coefficient for the output result with the same input. To use it, after compiling it with 'gcc diffusivity.c -o cal_diffusivity -lm', use './cal_diffusivity example.input output.dat' and the diffusivity coefficient would be printed.