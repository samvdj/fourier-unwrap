# fourier-unwrap
Fourier-based phase unwrapping 

The compressed .zip files contain executables which perform graphics processing unit (GPU)-supported Fourier-based phase unwrapping as described in the paper "Fast Fourier-based phase unwrapping on the graphics processing unit in real-time imaging applications" by Sam Van der Jeught et al. (2015). The executable has been precompiled to operate on various input map pixel sizes. Randomly generated wrapped phase maps for every pixel size are included in each folder.The user can replace the 'Image_Wrapped.bin' file with its own wrapped phasemap to test the robustness and speed of the algorithm.

The software requires the CUDA 6.0 runtime (https://developer.nvidia.com/cuda-toolkit) or later to be installed. OpenCV (http://opencv.org/) is used for visualization and input images are accepted in binary (.BIN) format.

For further information please contact: sam.vanderjeught@uantwerpen.be
