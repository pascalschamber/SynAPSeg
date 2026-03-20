
# SynAPSeg
![logo](/SynAPSeg/UI/icons/SynAPSeg_logo.png)

SynAPSeg is a flexible Python image analysis framework for fully automated, deep learning-based detection and quantification of fluorescent microscopy data. 

While designed with synaptic analysis in mind, the platform is agnostic to specific experimental conditions and serves as a general-purpose tool for large-scale image analysis.

![overview graphic](assets/2026_0310_SynAPSeg__Figure%203.png)

## Features
* **Graphical User Interface:**  No coding experience needed.
* **Unified Model Integration:** Integrates modern deep learning models through a single interface.
* **Fully Automated Workflows:** Facilitates automated end-to-end processing, replacing labor-intensive manual steps with pipelines.
* **Multi-Dimensional Data Support:** Handles high-dimensional data, internally standardizing to STCZYX (Sample, Time, Channel, Z, Y, X) formats.
* **Broad Format Compatibility:** Supports a wide array of input image formats (e.g. .TIFF, .CZI, .VSI) via the Bio-Formats and AICSImageIO libraries.
* **Interactive Annotation:** Provides a Napari-based suite to visually verify data, manually refine segmentation masks, and draw custom Regions of Interest (ROIs) that automatically integrate into downstream workflows.
* **Advanced 3D Quantification:** Offers a robust suite of quantification methods to extract morphological features, perform object-based colocalization, and run spatial distribution analyses.
* **Open Data Management:** Generates human-readable metadata and employs a project-based structure, allowing external tools to interface seamlessly for tasks like brain atlas registration.
* **Plugin-Style Framework:** Developers can easily incorporate new deep learning models or quantification techniques with minimal code.
* **Multi-Platform Support:** Tested across Windows, Mac (Apple silicon), and Linux operating systems.

## Core modules
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



## Installation instructions

### Prerequisites
1) conda/miniconda:
    * [Link](https://docs.conda.io/en/latest/miniconda.html)
2) For GPU support:
    * If using a GPU, ensure your system meets the hardware requirements
    * e.g., for Windows: Microsoft Visual C++ Redistributable, NVIDIA drivers, etc.
    * TensorFlow provides a good guide for [System requirements](https://www.tensorflow.org/install/pip)
3) download repository from GitHub
    * https://github.com/pascalschamber/SynAPSeg.git
    * or if you have git installed
        ```bash
        git clone https://github.com/pascalschamber/SynAPSeg.git
        ```


### Setup Instructions
Complete the below steps in your terminal

1. Navigate to the directory where the repository was downloaded/cloned on your computer
```bash
cd ".\downloads\SynAPSeg"
```


2. Set up the conda environment based on your hardware. Choose **one** of the following:
```bash
conda env create -f synapseg_conda_env_gpu.yaml 
# OR 
conda env create -f synapseg_conda_env_cpu.yaml 
# OR 
conda env create -f synapseg_conda_env_mac.yaml 
```


3. Activate the environment and install the package:
```bash
conda activate synapseg
pip install -e .
```


4. Run the user interface (initial setup may take a minute):
```bash
python -m SynAPSeg
```

More more information see the full video demo: 
* **coming soon**

---

## Third-Party Code & Acknowledgments

In addition to the above mentioned libraries, this project utilizes code modified from the following open-source repositories:

* **[ome-tiff-pyramid-tools](https://github.com/labsyspharm/ome-tiff-pyramid-tools)**: Developed by the Laboratory of Systems Pharmacology at Harvard Medical School. 
    * **License**: MIT License
    * **Usage**: Modified for writing ome.tiff image pyramids


## How to cite
* "SynAPSeg: A novel dataset and image analysis framework for deep learning-based synapse detection and quantification"
Pascal Schamber, Sahana Darbhamulla, Molly Boyer, Madison Pelletier, Helene Hartman, Olivia Friedman, Shiyu Zhang, Allison Blais, Seyun Oh, Haining Zhong, Alexei M Bygrave
bioRxiv 2026.03.12.711395; doi: https://doi.org/10.64898/2026.03.12.711395

