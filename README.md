# Sketch2Voxels
Sketch based modeling using deep convolution neural network.

This Project referred the work 'MarrNet: 3D Shape Reconstruction via 2.5D Sketches' by Jiajun Wu etc.

My code based on Tensorflow1.3 and Torch7.

Here are some results:


![](https://github.com/Jhonve/Sketch2Voxels/raw/master/Results/results.png)


Run the GUI firstly
```Shell
python3 SketchModeling.py
```

Then run the SketchGAN and the VoxelGAN
```Shell
cd SketchGAN/ScribblerNet
python3 main.py
cd ../../VoxelGAN/
th test_one.lua
```
