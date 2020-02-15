#!/bin/bash
# Aim to develop a quick simple script for transferring experimental results
#arguments: 1 - base file path on other machine
# 2: array of experiment files
# will create copies of the experiment files with the same names
# in the directory the script is called in on the home machine.
HOSTNAME=$1
BASEPATH=$2
filenames=("$@")
for ((i=2; i<${#filenames[@]}; i++ ));
do
  connect_string="${HOSTNAME}:${BASEPATH}"
  scp -r "${connect_string}/${filenames[$i]}" ${filenames[$i]}
  #echo ${connect_string}
  #echo "${filenames[$i]}"
done
