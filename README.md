# FCH: Feature-refinement Classification Head

This repository provides a PyTorch implementation of **FCH (Feature-refinement Classification Head)**, a lightweight and modular classification head proposed in the paper:

**Improvement of Classification Models through Graph Neural Networks and Batchwise Relational Encoding**
Sooin Kim, Kyungtae Kim, Donghoon Kim, Doosung Hwang

FCH enhances image classification by modeling **inter-sample relationships within each mini-batch** using a Graph Neural Network (GNN). It refines backbone feature embeddings through batchwise graph construction and joint optimization of classification and structure-preserving losses.

---

## Key Ideas

* Treats samples in a mini-batch as nodes in a graph
* Builds an adjacency matrix from feature similarities (RBF, k-NN, Attention, or Class-Label based)
* Applies a GNN layer to refine embeddings using relational information
* Optimizes a joint objective:

  * Cross-entropy loss for classification
  * Structure-preserving loss to align learned relations with label structure

---

## Architecture Overview

1. **Backbone Feature Extractor**
   A standard CNN or Transformer (e.g., ResNet-18, ViT-B/16) extracts D-dimensional embeddings.

2. **Batchwise Graph Construction**
   Each mini-batch is treated as a graph, where nodes are samples and edges are defined by feature similarity.

3. **Graph Neural Network Layer**
   A Chebyshev polynomial-based graph convolution aggregates neighborhood information to refine features.

4. **MLP Classifier**
   A two-layer MLP maps refined embeddings to class probabilities.

---

## Supported Adjacency Types

* **RBF-FCH**: Radial Basis Function kernel on feature distances
* **NN-FCH**: k-nearest neighbor graph with RBF weights
* **Att-FCH**: Dot-product attention with neighborhood restriction
* **CL-FCH**: Binary graph using class-label consistency

---

## Installation

```bash
conda create -n fch python=3.9
conda activate fch
pip install -r requirements.txt
```

---

## Training Example

```bash
python train.py \
  --dataset cifar10 \
  --backbone resnet18 \
  --fch_type rbf \
  --chebyshev_p 1 \
  --batch_size 128 \
  --lr 1e-4
```

---

## Experimental Results (Quoted from the Paper)

### Classification Performance (Accuracy)

| Model                        | CIFAR-10  | MNIST     | STL-10    |
| ---------------------------- | --------- | --------- | --------- |
| EfficientNetV2-L             | 0.991     | 0.998     | 0.950     |
| DeiT-B                       | 0.991     | 0.997     | 0.948     |
| VGG-5                        | 0.904     | 0.990     | 0.940     |
| Att-FCH (ResNet-18, p=5)     | 0.963     | 0.993     | 0.910     |
| NN-FCH (ResNet-18, p=4)      | 0.972     | 0.994     | 0.915     |
| CL-FCH (ResNet-18, p=1)      | 0.971     | 0.994     | 0.914     |
| **RBF-FCH (ResNet-18, p=1)** | **0.979** | **0.995** | **0.922** |
| Att-FCH (ViT-B/16, p=5)      | 0.977     | 0.996     | 0.948     |
| NN-FCH (ViT-B/16, p=4)       | 0.977     | 0.996     | 0.950     |
| CL-FCH (ViT-B/16, p=1)       | 0.976     | 0.996     | 0.949     |
| **RBF-FCH (ViT-B/16, p=1)**  | **0.979** | **0.997** | **0.953** |

These results demonstrate that FCH:

* Consistently improves performance across datasets and backbones
* Provides strong gains on CIFAR-10 and STL-10
* Achieves competitive accuracy with far fewer parameters than large-scale models

---

## Model Complexity (from the Paper)

| Model                    | Params (MB) | GFLOPs |
| ------------------------ | ----------- | ------ |
| EfficientNetV2-L         | 454.05      | 12.41  |
| DeiT-B                   | 330.22      | 16.87  |
| VGG-5                    | 81.55       | 19.63  |
| RBF-FCH (ResNet-18, p=1) | 43.00       | 4.22   |
| RBF-FCH (ViT-B/16, p=1)  | 333.75      | 15.52  |

FCH (ResNet-18) delivers competitive accuracy with only ~43 MB parameters and ~4.2 GFLOPs.

---

## Citation

If you use this code or FCH in your research, please cite:

```
@article{kim2024fch,
  title   = {Improvement of Classification Models through Graph Neural Networks and Batchwise Relational Encoding},
  author  = {Kim, Sooin and Kim, Kyungtae and Kim, Donghoon and Hwang, Doosung},
  journal = {---},
  year    = {2024}
}
```

---

## License

This project is released under the MIT License.

---
