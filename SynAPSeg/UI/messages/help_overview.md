# overview
This package is designed to be a flexible framework for automated fluorescent microscopy image analysis of neural data.
We provide a high-throughput workflow that allows for segmentation, colocaliziation, registration, morphology extraction, quantification, and visualization of neural structures. 
It's primary purpose is to integrate modern deep learning models to segment synapses and dendrites, provide a user interface for annotating data, and wrap this all into a pipeline to quantify and analyze the results.
Though usage is targeted toward segmentation/quantification of synaptic structures we also apply this tool to perform longitudinal tracking, nuclei detection, and provide a interface to align data to brain atlases using the ABBA package.
We provide a synapse segmentation model that has been trained on over 40,000 manually annotated synapses across imaging modalities and using a wide variety and associated training data available here: 
The core pipeline consists of the following features
    1. Segmentation
        supports many common image formats as input
        applies ML models for self-supervised denoising (N2V) and segmentation of synapses and dendrites with custom Stardist and Unet models
    2. Annotation
        We utilize Napari to provide a GUI to verify segmentation results, allow manual annotations, and defining ROIs.
    3. Quantification
        We have implemented many approaches for feature extraction:
            object counts, object morphology (area, intensity), object colocalization, spatial localizaiton (distance thresholding, distance from soma), ROI extraction, and supports extracting assignments based on manually defined ROIs.
    4. Analysis
        We provide a generic method to generate a standard array of graphical analyses and a foundation for more customized analyses.