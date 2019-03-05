# MFIN: Motion Field Interpolation Network
Temporal Interpolation via Motion Field Prediction

This repository contains the tensorflow implementation for the MFIN network presented at MIDL 2018 in Amsterdam.

If you use the code, please cite our paper

Temporal Interpolation via Motion Field Prediction <br>
Authors: Lin Zhang*, Neerav Karani*, Christine Tanner, Ender Konukoglu <br>
MIDL 2018. [arXiv:1804.04440](https://arxiv.org/abs/1804.04440)

An Image Interpolation Approach for Acquisition Time Reduction in Navigator-based 4D MRI <br>
Authors: Neerav Karani*, Lin Zhang*, Christine Tanner, Ender Konukoglu <br>
Medical Image Analysis 2019. [https://doi.org/10.1016/j.media.2019.02.008](https://doi.org/10.1016/j.media.2019.02.008)



# Instruction
An abdominal MR image sequence extracted from our test dataset is provided as example. To run the example, please use the following command 

```
python test_mfin_cycle.py ssim
```
for the pre-trained model with SSIM based loss function, or
```
python test_mfin_cycle.py l2
```
for the pre-trained model with L2 norm based loss function.

To visualize the results, please use the matlab script ```show_results.m```.
