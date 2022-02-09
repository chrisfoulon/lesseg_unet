#!/usr/bin/env bash

display_usage() {
	echo "DDP lesseg_unet launcher"
	echo -e "\nUsage: $0 number_of_GPUs [lesseg_unet arguments ...] \n"
	echo "lesseg_unet parameters:"
	lesseg_unet -h
}

if [[ ( $# == "--help") ||  $# == "-h" ]]
then
		display_usage
		exit 0
fi

if [  $# -le 4 ]
then
		display_usage
		exit 1
fi


args=("$@")
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${args[0]}" \
    ./lesseg_unet/main.py "-h"
#    ./lesseg_unet/main.py "${args[@]:1}"
#    lesseg_unet "${args[@]:1}"