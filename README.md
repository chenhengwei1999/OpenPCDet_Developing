# Developing of OpenPCDet

This repository forked from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), aiming to develop some algorithms and document concise and pratical tutorials, so as to use it simply.

## Running Environment

* Ubuntu 20.04

* OpenPCDet v0.6.0

## Introduction

Official document can be referred to [here](./README_official.md).

### What does `OpenPCDet` toolbox do?

`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

Based on `OpenPCDet` toolbox, [OpemMMLab](https://github.com/open-mmlab) win the Waymo Open Dataset challenge in [3D Detection](https://waymo.com/open/challenges/3d-detection/), [3D Tracking](https://waymo.com/open/challenges/3d-tracking/), [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/) 
three tracks among all LiDAR-only methods.

### `OpenPCDet` design pattern

- Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:

<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

Here are a few thoughts (questions) in response to this pattern?

1. How do they realize the separation of data and model? What are the shortcomings in previous or other works?

    Simply speaking, the separation refers to decompart the codes of building dataset loader and creating model networks. Take concrete codes as an example, i.e. [separation_of_data_and_model.ipynb](docs/introduction/separation_of_data_and_model.ipynb). Fast training and testing of datasets such as `kitti`, `nuscenes`, `waymo`, etc., can be achieved through the interface provided by the `pcdet` toolkit, and the network parameters can also be modified by modifying the `yaml` file prior to training, a programming specification which is of great help to us. In the `tools` directory, [demo.py](./tools/demo.py) is fine example, by inheriting `DatasetTemplate`, it is possible to read and infer sample data from different datasets.

2. What is the unified point cloud coordinate, and why are they doing it?

    [#Issue 236](https://github.com/open-mmlab/OpenPCDet/issues/236) raised an question about *"How does deal with the ".pcd" point cloud files to the unified normative coordinate of OpenPCDet?"*. With the help of this question, it is just necessary to understand what the unified point cloud coordinates are in the `pcdet` toolbox.

3. How does this toolbox extent to custom datasets, such as Carla dataset or physical vehicle's dataset?

    ...

