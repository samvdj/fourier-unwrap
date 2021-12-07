# fourier-unwrap
Fourier-based phase unwrapping 

Here, we present a high-speed phase unwrapping algorithm to unwrap images that are mathematically wrapped to the finite interval [−π, π], corresponding to the principle value domain of the arctangent function. By executing the parallel implementation of a single-step Fourier-based phase unwrapping algorithm on the graphics processing unit of a standard graphics card, the total processing time of the phase unwrapping algorithm is limited to <5 ms when executed on a 640 × 480-pixel input map containing an arbitrarily high density of phase jumps. In addition, we expand upon this technique by inserting the obtained solution as a preconditioner in the conjugate gradient technique. This way, phase maps that contain regions of low-quality or invalid data can be unwrapped iteratively through weighting of local phase quality.

The compressed .rar files contain executables which perform graphics processing unit (GPU)-supported Fourier-based phase unwrapping as described in the paper "Fast Fourier-based phase unwrapping on the graphics processing unit in real-time imaging applications" by Sam Van der Jeught et al. (2015). The executable has been precompiled to operate on various input map pixel sizes. Randomly generated wrapped phase maps for every pixel size are included in each folder.The user can replace the 'Image_Wrapped.bin' file with its own wrapped phasemap to test the robustness and speed of the algorithm.

The Functions.cpp and Functions.h files contain helper functions, the kernel.cu file contains the bulk of the Fourier-based unwrapping algorithm.

The software requires the CUDA 6.0 runtime (https://developer.nvidia.com/cuda-toolkit) or later to be installed. OpenCV (http://opencv.org/) is used for visualization and input images are accepted in binary (.BIN) format.

For further information please contact: samvdj@gmail.com
