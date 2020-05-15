# Localization - Particle Filter
---
The goals / steps of this project are the following:
* Initialize reasonable particles based on GPS data using a Gaussian distribution
* Predict the motion model using a bicycle model in discrete timesteps
* Use Nearest Neighbor algorithm to associate predictions with the observations
* Update weights using the multi-variate Guassian distribution
* Re-sample all particles based on particle weights
* Test the algorithms in a 2D car simulator with landmarks


[//]: # (Image References)

[image1]: report_images/Sim1.JPG

---
## Project Overview
This project utilizes a particle filter to localize a 2D vehicle based on landmarks, with the inclusion of gaussian noise and motion models.

Inputs:
* Map in global coordinates
* Noisy GPS estimate for initialization
* Noisy sensor/control data

Scripts (in /src):

* `main.cpp` - communicates with the simulator, inputs sensor/motion measurements and initial GPS location, runs all other scripts
* `helper_function.h` - contains functions and structures for the particle filter
* `map.h` - list of landmarks with (x,y) locations in global coordinates
* `particle_filter.cpp` - all particle filter algorithms, including initialization, weight updates, resampling, and the nearest neighbor algorithm

My work is in the `particle_filter.cpp`.

An example of the output is shown below (blue circle is my localization and the car is the ground truth):

![][image1]

The resultant accuracy is ~10cm and under 0.2 degrees in yaw. 

---
## Try it yourself!
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. ./clean.sh
2. ./build.sh
3. ./run.sh
