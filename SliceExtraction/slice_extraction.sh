#!/bin/bash
script_dir=$(realpath $(dirname $0))
data_path=#Add here path to data
model_path=#Add here path of your model, e.g. /home/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task100_SliceExtraction/nnUNetTrainerV2__nnUNetPlansv2.1
qcModelPath=$script_dir/QualityControlModel/LogisticRegression_QC_sliceExtraction.pkl

cd $script_dir/run &&
python run_prediction.py -d $data_path --model_path $model_path --modality CT --visualize --qcModelPath $qcModelPath --wantedSpacing 2 2 5 --applyQC --changeSpacing

