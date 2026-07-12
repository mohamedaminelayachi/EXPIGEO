

![EXPIGEO Comparative Examples](assets\EXPIGEOLogoWhite.png)

This repository contains the official implementation of EXPIGEO (pronounced [**/ˌɛksˈpɛdʒi.oʊ/**](https://ipa-reader.com/?text=%2F%CB%8C%C9%9Bks%CB%88p%C9%9Bd%CA%92i.o%CA%8A%2F)) introduced in the MICCAI 2026 paper titled "Exploiting Interior Geometry for Intracranial Aneurysm Detection and Segmentation on 3D Point Clouds via GNNs." EXPIGEO is a framework that proposes novel geometry-based techniques used to extract physiologically meaningful geometric descriptors that unlock GNNs' performance on Intracranial Aneurysm Detection and Segmentation on 3D Point Clouds.

Authors: Mohamed Amine Layachi, Anass Nouri, Souad Eddarouich, and Florent Autrusseau.


## Abstract

The automated 3D detection and segmentation of intracranial aneurysms is critical for clinical intervention, yet existing deep learning methods often struggle with the complex, tubular organization of cerebral vasculature. We propose EXPIGEO, a geometry-aware framework that bridges this gap by integrating explicit geometric reasoning with topological graph representation. Standard local features are easily confounded by the inherent curvilinearity of the vasculature, heavily limiting the effectiveness of conventional Graph Neural Networks (GNNs). EXPIGEO overcomes this by introducing an internal geometric exploration mechanism to characterize the vessel interior, enabling the robust extraction of skeletal topology and physiological descriptors. This explicit structural encoding successfully resolves topological ambiguities and unlocks GNN performance, establishing a new state-of-the-art benchmark for aneurysm detection on the IntrA dataset with 98.41% accuracy and an F1-score of 0.9589. Beyond raw performance, EXPIGEO provides clinical transparency: instance-level explainability confirms that predictions are driven by biologically relevant volumetric features, ensuring a reliable and interpretable diagnosis. 

## An Intuitive Explainer on Interior Geometry Exploration

<video width="640" height="360" controls>
  <source src="assets\EXPIGEO_Manimated.mp4" type="video/mp4">
  An Intuitive Explainer on Interior Geometry Exploration
</video>

## Visuals

![EXPIGEO Comparative Examples](assets\EXPIGEOVisual.png)

## Usage

There are four steps when using the system: Building, Training, Testing, and Explaining.

- **Building**: used to build the feature-rich dataset using EXPIGEO and preparing it to be used by the GNN model in Training, Testing, and Explaining.

- **Training**: used to train the GNN model given the dataset from the dataset building stage.

- **Testing**: used to test the trained GNN given a model state--weights--and the test subset of IntrA.

- **Explaining**: used to perform a post-hoc explainability analysis for the trained GNN model.

### Building

In order for the GNN to use EXPIGEO efficiently, we need to build the feature-rich dataset before running any other step. When we say feature-rich, we mean the dataset where each point cloud also includes all the geometric descriptors presented in the paper. This process ensures the training speed of the GNN to be as fast as possible. Therefore, we advise the user to build the dataset separately then proceed with the other steps.

**Please Note** that the speed at which the dataset is being built will depend on the point density (number of points) and the CPU of the machine. To be efficient, we recommend the user to simplify the input point clouds into a point density of about 512, 1024, or 2048. One can aim for higher, but they will trade-off speed in the process.

**Also Note** that one can use the core script (`expigeo.py`) independently if they want to integrate it into their pipeline. The result will always be a feature-rich point cloud.


To build the feature-rich dataset, use this command:

```
python main.py build -rd IntrA -s IntrABuild -t balanced -exp expigeo/expigeo_params.yaml
```

Where,

- `build` is the command for the building process.
- `-rd` is an argument for the root directory of the IntrA dataset. In our case, it's just ```IntrA```.
- `-s` is an argument for the save path of the feature-rich dataset (e.g. ```IntrABuild```).
- `-t` is an argument used to specify which files from the dataset the user is targeting (`balanced` for a balanced between aneurysmal and healthy vessel samples, `positives` only, `negatives` only, `rem_negatives`, `all_files` for all samples in IntrA). Default is `balanced`. `rem_negatives` is not a practically useful option as it's used only when sampling a balanced set from all the files (IntrA comes with more negatives than positives), but it's worth keeping it there for users should they find a utility for it.
- `-exp` is an argument for a YAML file that is used to configure all the parameters of EXPIGEO. It's a good practice to start with the predefined file to get a baseline, then fine-tune the parameters based on the user's case.

### Training

Once the feature-rich dataset is built, the user can use it to start training the GNN model.


To train the GNN model, use this command:

```
python main.py train -cfg expigeo/train_config.yaml
```

Where,

- `train` is the command for the training process.
- `-cfg` is an argument for the training configuration YAML file (Check the predefined config file).

**Please Note** that there's a model config file (`model_config.yaml`) referenced in the training config file, which configures the architecture of the GNN model.

### Testing

Once the model is trained, the user can test it.


To test the GNN model, use this command:

```
python main.py test -cfg expigeo/test_config.yaml
```

Where,

- `test` is the command for the testing process.
- `-cfg` is an argument for the testing configuration YAML file (Check the predefined `test_config.yaml` file).

**Very Important:** 

1) There's an argument in the test config file named `data_state`, which is crucial to use in order to prevent data leakage in testing. It's automatically and implicitly saved (`data_state.json`) during training to know what samples were used in training and validation to not use them in testing.

2) There's an argument in the test config file named `model_state`, which is the path to the model checkpoint (`.pth`) the user wants to use during testing. Make sure to select a valid checkpoint.


### Explaining

In deep learning, we often just a train, test, and deploy a model **without** inspecting it's internals (which is a complex process, and there's a whole field on interpretability focused solely on this). In medical imaging, however, we shouldn't follow this path as best as we can because the risks are higher; telling a patient that they have an aneurysm when they don't is as risky if they do. Hence, knowing the drivers behind a model's predictions is as important if not more than the predictions themselves. Luckily, research on interpretability is accelerating quite fast.

Inspired by the accelerated progress, we provide two main explanatory quantities that corroborates the physiological-alignment of our GNN model and, hopefully, can lead to more clinical insights going forward. These two quantities are: the Receptive Field Entropy ($H_{RF}(p)$) and Feature Importance. Intuitively, $H_{RF}$ reflects the disorder in the signals within the receptive field of the GNN at a particular node (p); meaning, if signals are conflicted such that some nodes' are classified to be aneurysmal while others are healthy in the same receptive field of $p$, then the entropy will be higher since these decisions lack certainty. This implies that features are not strong enough to overcome this entropy. Whereas, the feature importance quantifies and determines what features are helpful and important to the model in order to predict a certain class for a point.


To test the GNN model, use this command:

```
python main.py explain -cfg expigeo/test_config.yaml
```

Where,

- `test` is the command for the explain-process.
- `-cfg` is an argument for the testing configuration YAML file (Same as the one used in test).

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{Layachi_EXPIGEO_MICCAI2026,
  author={Layachi, M.A. AND Nouri, A. AND Eddarouich, S. AND Autrusseau, F.},
  title={Exploiting Interior Geometry for Intracranial Aneurysm Detection and Segmentation on 3D Point Clouds via GNNs},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026},
}
```

If there are questions about this work or any issues with the implementation, please do not hesitate to contact us.