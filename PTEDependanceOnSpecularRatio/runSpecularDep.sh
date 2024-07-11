#!/bin/bash

# Define the total number of steps
steps=101

# Iterate over the range and run the Python script
for ((i = 0; i < $steps; i++)); do
    echo "Running step $i of $steps"
    python SpecularDependance.py $steps $i
done
