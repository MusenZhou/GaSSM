#!/bin/bash -l


if [ -d "/usr/local/cuda" ]
then
	echo "nvcc may exist"
else
	echo "warning!!! nvcc may not exist"
fi


echo "try to compile"


nvcc ./source_code/RC_main.cu -I ./source_code/debug.h -I ./source_code/cpu_fxn.h -I ./source_code/device_fxn.h -I ./source_code/global.h -o GPU_string_polyatmoic -lm -w




if [ -f "GPU_string_polyatmoic" ]
then
	echo "success!"
else
	echo "FAILED!!!"
fi