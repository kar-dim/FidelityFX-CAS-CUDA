# FidelityFX-CAS-CUDA

<p align="center">
<img src="https://github.com/user-attachments/assets/06eaafc2-7bfa-4bff-ab48-646230ddd936"></img>
</p>

[Contrast Adaptive Sharpening (CAS)](https://gpuopen.com/fidelityfx-cas/) is a low overhead adaptive sharpening algorithm with optional up-sampling. The technique is developed by Timothy Lottes (creator of FXAA) and was created to provide natural sharpness without artifacts.

It is used in 3D Graphics frameworks like DX12 and Vulkan, and provides a mixed ability to sharpen and optionally scale an image. **This project implements only the sharpening part**. The algorithm adjusts the amount of sharpening per pixel to target an even level of sharpness across the image. Areas of the input image that are already sharp are sharpened less, while areas that lack detail are sharpened more. This allows for higher overall natural visual sharpness with fewer artifacts. CAS was designed to help increase the quality of existing Temporal Anti-Aliasing (TAA) solutions. TAA often introduces a variable amount of blur due to temporal feedback. The adaptive sharpening provided by CAS is ideal to restore detail in images produced after TAA.
<br></br>

<p align="center">
<img src="https://github.com/user-attachments/assets/670b2932-8c3c-4e6d-88ee-be4f5dae2d28"></img>
</p>

This project implements CAS as a CUDA kernel. The main reasons for porting CAS to CUDA are:
1. General purpose. Because CAS is technically a filter, it can also be used for sharpening static images (like local files from disk). The original CAS filter works only in 3D graphics frameworks.
2. Speed. By implementing the CAS algorithm efficiently in CUDA, we can expect major speedups compared to CPU implementations by leveraging the GPU's high performance in parallel problems.

Τhis repository has two projects:

1. **CAS Implementation**. The core functionality of CAS, implemented as a CUDA kernel. It is a DLL project, and defines a C-style interface for interacting with the DLL. Programs that will use CAS filtering, should have in the output directory the CAS ```.dll``` and ```.lib``` files, and also include the ```CASLibWrapper.h``` file to interact with the DLL.
2. **GUI Application**. This simple GUI project aims to showcase how to interact with the CAS DLL in order to sharpen images.

## Build

The projects are included in a Visual Studio Solution (```.sln```).
1. CAS DLL requires CUDA Toolkit (tested with version 12.6) in order to link with the CUDA libraries and to include the CUDA header files.
2. The Qt GUI application requires Qt MSVC (tested with version 6.8.0) in order to use the Qt framework.
3. When building the GUI project, the tool ```windeployqt``` is called in order to copy the required Qt dependencies for running the application. Also, the DLL is copied in the GUI application's output folder.


## GUI Application usage

1. Launch the application.
2. Use the **Open Image** from the File menu to select an image file from the system.
3. Adjust parameters as desired through the user interface.
4. The sharpening is applied in realtime each time a parameter is changed, to allow the user to view the updated image with various configurations.
5. (Optional) Save the processed image using the **Save Image** from the File menu.

## Prerequisites/Dependencies

- **NVIDIA GPU**: An NVIDIA GPU is required in order to use the CAS DLL.
