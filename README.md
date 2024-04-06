# EgoTAP - CVPR 2024 Highlight
Official repository of the "Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting"

## [arXiv](https://arxiv.org/abs/2402.18330) / [CVPR Open Access](https://openaccess.thecvf.com/menu)
Links will be updated as they are uploaded.

https://github.com/tho-kn/EgoTAP/assets/54742258/fc6a1f4d-3df9-4f96-a755-067468843c40

## Citation
```
@InProceedings{kang2024egotap,
    author    = {Kang, Taeho and Lee, Youngki},
    title     = {Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}
```

## Datasets
### EgoCap
The full dataset is not publicly available. If you get access to the full dataset, extract `EgoCapDataloader3D` in this repository's root. you should also download the publicly available dataset together. It should be organized as follows

        /path/to/egocap
        ├── training_v000
        │   ├── ...
        ├── training_v002
        │   ├── ...
        ├── validation_v003_2D
        │   ├── ...
        ├── validation_v003_3D
        │   ├── ...

Use the following script to process the dataset for our code.

        python reprocess_egocap_data.py --data_dir /path/to/processed_egocap --metadata_dir /path/to/egocap --joint_preset EgoCap

Following [Ego3DPose](https://github.com/tho-kn/Ego3DPose), the non-augmented part of the dataset is used. The dataset is processed with the following script, to make it compatible with our code.

### UnrealEgo
After downloading the UnrealEgo dataset following the instructions in [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo) repository, reprocess the dataset for our code.
```reprocess_unrealego_data.py``` parses metadata and adds some additional 2D and 3D data to data pickle.

        python reprocess_unrealego_data.py --data_dir /path/to/processed_unrealego --metadata_dir /path/to/unrealego

## Implementation

### Dependencies 
Our code is tested in the following environment

- Python 3.8.10
- Ubuntu 20.04
- PyTorch 2.0.1
- CUDA 12.0

You can install other required packages with requirements.txt

### Training

You can train the models from scratch or use [trained weights](https://drive.google.com/drive/folders/1l5DsnC8jtlyxXGveNsTcC744mPn_oFh1?usp=sharing). The weights are re-trained using this refactored code. The model weights will be saved in the `log_dir` specified in `options/constants.py`. The provided weights should be placed in the same directory.

#### Heatmap Estimator

        bash scripts/train/Heatmap/Joint/unrealego.sh
        bash scripts/train/Heatmap/Limb/unrealego.sh

Please specify the path to the UnrealEgo dataset in '--data_dir'.

Use scripts named egocap.sh to train for the EgoCap dataset.
The hasty convergence to zero is more frequently observed, so it might require manual retraining.
        
#### Pose Estimator

        bash scripts/train/PoseEstimator/egocap.sh
        bash scripts/train/PoseEstimator/unrealego.sh

please specify the path to the dataset in '--data_dir'.
After the training is finished, you will see quantitative results.

### Testing

If you want to see quantitative results using trained weights, run the command below.
This will also output result summary as a text file, which can be used for ploting for comparison of methods.

        bash scripts/test/EgoTAP/egocap.sh
        bash scripts/test/EgoTAP/unrealego.sh

You can compare result to the baseline methods in the [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo) repository and [Ego3DPose](https://github.com/tho-kn/Ego3DPose) repository. The dataset processing code is mostly compatible.

## License Terms
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments
This code is based on [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo) repository, and thus inherited its license terms.
We thank the authors of the UnrealEgo for the permission to share our codes.
