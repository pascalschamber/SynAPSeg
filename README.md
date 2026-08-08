## Overview
SynAPSeg is a flexible Python image analysis framework for fully automated, deep learning-based detection and quantification of fluorescent microscopy data.
While designed with synaptic analysis in mind, the platform is agnostic to specific experimental conditions and serves as a general-purpose tool for large-scale image analysis.


#### Quick links
* [manuscript](https://doi.org/10.64898/2026.03.12.711395)
* [analysis code](https://github.com/pascalschamber/SynAPSeg_manuscript_code)
* [SynAPSeg dataset/pre-trained models](https://zenodo.org/records/18988899)



## Tutorial
We provide a full walkthrough of how to install and use SynAPSeg in the following video tutorial:

[![getting started tutorial](https://i.ytimg.com/vi/FDJTJSOGUt0/hqdefault.jpg?sqp=-oaymwEnCNACELwBSFryq4qpAxkIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB&rs=AOn4CLAkkq_G6US2CtaBKlOwndtl9wAe0Q)](https://youtu.be/FDJTJSOGUt0 "getting started tutorial")


## Installation

### Prerequisites
* [Conda/miniconda](https://docs.conda.io/en/latest/miniconda.html) or another python package manager:
* If using a GPU, ensure your system meets the hardware and software requirements. TensorFlow provides a good guide of the [system requirements](https://www.tensorflow.org/install/pip#hardware_requirements).


### Setup Instructions

1) Download the [repository](https://github.com/pascalschamber/SynAPSeg/archive/refs/heads/main.zip) from GitHub, unzip, and save to desired location.

*complete the rest of the steps in your computer's terminal application*

2) Navigate to the directory where the repository was downloaded/cloned on your computer, for example:  
     
    ```bash
    cd "C:\Program Files\SynAPSeg-main"
    ```
    *note: this is just an example, the path will depend on where you saved it*


3) Set up the conda environment based on your hardware. Run **one** of the following:
    ```bash
    # for windows/linux with GPU
    conda env create -f synapseg_conda_env_gpu.yaml

    # without GPU 
    conda env create -f synapseg_conda_env_cpu.yaml 

    # for mac 
    conda env create -f synapseg_conda_env_mac.yaml 
    ```


4) Activate the conda environment that was just created:
    ```bash
    conda activate synapseg
    ```

5) Install the package 
    ```bash
    pip install -e .
    ```

6) Run the user interface (initial setup may take a minute):
    ```bash
    python -m SynAPSeg
    ```


### To run the application in the future:
* open your terminal and activate the environment `conda activate synapseg`
* run the command `python -m SynAPSeg`

<br>


## Core modules

<img src="assets/2026_0310_SynAPSeg__Figure%203.png" alt="overview graphic" height="50%"/>

<br>



The framework is structured around three core stages:


1) Segmentation:

    Allows models to be chained together (e.g. denoising model --> segmentation model)

    Use weights of pre-trained, our [synapse detection models](https://zenodo.org/records/18988899), or your own custom trained models

    Currently supports following models: 
    * [Stardist](https://github.com/stardist/stardist)
    * [N2V/N2V2 via Careamics](https://github.com/CAREamics/careamics)
    * [Cellpose](https://github.com/MouseLand/cellpose)
    * [Segmentation Models (e.g. U-Net models)](https://github.com/qubvel/segmentation_models)

2) Annotation
    
    Utilizes [Napari](https://github.com/napari/napari) to provide a GUI to verify segmentation results, perform manual refinement, and defining ROIs.
    Provide a suite of interactive widgets to facilitate these tasks.

3) Quantification
    
    We have implemented many approaches for feature extraction:
    * object counts, object morphology (size, intensity, etc.) 
    * object colocalization
    * spatial localizaiton (ROI extraction)

## Features
* **Graphical User Interface:**  No coding experience needed.
* **Multi-Platform Support:** Tested across Windows, Mac (Apple silicon), and Linux operating systems.
* **Broad Format Compatibility:** Supports a wide array of input image formats (e.g. .TIFF, .CZI, .VSI) via the Bio-Formats and AICSImageIO libraries.
* **Fully Automated Workflows:** Facilitates automated end-to-end processing, replacing labor-intensive manual steps with pipelines.
* **Unified Model Integration:** Integrates modern deep learning models through a single interface.
* **Interactive Annotation:** Provides a Napari-based suite to visually verify data, manually refine segmentation masks, and draw custom Regions of Interest (ROIs) that automatically integrate into downstream workflows.
* **2D/3D Quantification:** Offers a robust suite of quantification methods to extract morphological features, perform object-based colocalization, and run spatial distribution analyses.
* **Multi-Dimensional Data Support:** Handles high-dimensional data, internally standardizing to STCZYX (Sample, Time, Channel, Z, Y, X) formats.
* **Open Data Management:** Generates human-readable metadata and employs a project-based structure, allowing external tools to interface seamlessly for tasks like brain atlas registration.
* **Plugin-Style Framework:** Developers can easily incorporate new deep learning models or quantification techniques with minimal code.





## Third-Party Code & Acknowledgments

In addition to the above mentioned libraries, this project utilizes code modified from the following open-source repositories:

* **[ome-tiff-pyramid-tools](https://github.com/labsyspharm/ome-tiff-pyramid-tools)**: Developed by the Laboratory of Systems Pharmacology at Harvard Medical School. 
    * **License**: MIT License
    * **Usage**: Modified for writing ome.tiff image pyramids


## How to cite
* "SynAPSeg: A novel dataset and image analysis framework for deep learning-based synapse detection and quantification"
Schamber P, Darbhamulla S, Boyer M, Pelletier M, Hartman H, et al. (2026) SynAPSeg: A novel dataset and image analysis framework for deep learning-based synapse detection and quantification. PLOS Computational Biology 22(7): e1014571. https://doi.org/10.1371/journal.pcbi.1014571

