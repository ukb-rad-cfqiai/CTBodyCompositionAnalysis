# CTBodyCompositionAnalysis
![Fig1](https://github.com/ukb-rad-cfqiai/CTBodyCompositionAnalysis/assets/98951773/e2fcf0b3-7f0b-48bd-9ddc-747ee9a8db48)

# Installation
Please use python version 3.9.
For the slice extraction model, you need to install nnU-Net (Version 1).
Please follow the instructions at https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1 and install the pipeline as an **integrative framework**.

Then adapt the nnU-Net code to add the quality control method for slice extraction by copying the python scripts in /CTBodyCompositionAnalysis/SliceExtraction/nnUNet\_code folder (**predict.py** and **predict_simple.py**) to /...data\_path\_nnunet.../nnunet/inference.

Run
```
pip install -r requirements.txt
```
to install the necessary libraries.

# Usage
## Slice Extraction

Run the slice extraction by:
```
cd /.../CTBodyCompositionAnalysis/SliceExctraction
bash slice_extraction.sh
```
Adapt *data\_path* and *model\_path* in run\_pred.sh.

## Tissue Segmentation
Run the tissue segmentation by:
```
cd /.../CTBodyCompositionAnalysis/TissueSegmentation
python tissue_segmentation.py --input_path /path/to/L3L4Slices.nii.gz
```

The trained model is currently not publicly available due to German data protection law. Sharing the models requires a research agreement and board approval. Please contact sprinkart@uni-bonn.de 

# References
If you use this code, please cite
> 1. Nowak, S., Theis, M., Wichtmann, B.D. et al. End-to-end automated body composition analyses with integrated quality control for opportunistic assessment of sarcopenia in CT. Eur Radiol 32, 3142–3151 (2022). https://doi.org/10.1007/s00330-021-08313-x 
>2. Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
>3. Estrada, Santiago, et al. "FatSegNet: A fully automated deep learning pipeline for adipose tissue segmentation on abdominal dixon MRI." Magnetic resonance in medicine 83.4 (2020): 1471-1483. https://doi.org/10.1002/mrm.28022
