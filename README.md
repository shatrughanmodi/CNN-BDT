# CNN-BDT

This repository contains the code for approach named CNN-BDT as it uses CNN(Convolutional Neural Network) and BDT (Bagged Decision Tree). 

The CNN is used in PCE (Power Consumption Estimation) module, which estimates the power consumption of an EV based on seven different parameters:
1) Vehicle Speed
2) Vehicle Acceleration
3) Auxiliary Load
4) Road ELevation
5) Wind Speed
6) Environmental Temperature
7) Initial Battery's SOC

The CNN used in PCE Module is inspirod from the CNN architecture used for Hand Gesture Recognition in the article Deep Learning for Hand Gesture Recognition on Skeletal Data from G. Devineau, F. Moutarde, W. Xi and J. Yang. The code for CNN is written in Python using Pytorch API's.

The BDT is used as a Fine Tuner to fine tune the estimates given by PCE Module. The BDT is implemented using MATLAB 2019a.

Training the PCE Module:
1) Download the repository to your machine.
2) Run the Train_PCEModule.py file to train the CNN of the PCE Module.

Training Fine Tuner Module:

Run the code in Matlab to train a Bagged Decision Tree for fine tuning the output.
