# Converting High-Performance and Low-Latency SNNs through Explicit Modelling of Residual Error in ANNs

Codes for **Converting High-Performance and Low-Latency SNNs through Explicit Modelling of Residual Error in ANNs** in  IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

## Paper

**Abstract:** 

Spiking neural networks (SNNs) have garnered interest due to their energy efficiency and superior effectiveness on neuromorphic chips compared with traditional artificial neural networks (ANNs). One of the mainstream approaches to implementing deep SNNs is the ANN-SNN conversion, which integrates the efficient training strategy of ANNs with the energy-saving potential and fast inference capability of SNNs. However, under extreme low-latency conditions, the existing conversion theory suggests that the problem of SNNs neurons firing more or less spikes within each layer, i.e., residual error, leads to a performance gap in the converted SNNs compared to the original ANNs. This severely limits the possibility of practical application of SNNs on delay-sensitive edge devices. Existing conversion methods addressing this problem usually involve modifying the state of the conversion spiking neurons. However, these methods do not consider their adaptability and compatibility with neuromorphic chips. We propose a new approach based on explicit modeling of residual errors as additive noise. The noise is incorporated into the activation function of the source ANN, effectively reducing the impact of residual error on SNN performance. Our experiments on the CIFAR10/100 and Tiny-ImageNet datasets verify that our approach exceeds the prevailing ANN-SNN conversion methods and directly trained SNNs concerning accuracy and the required time steps. Overall, our method provides new ideas for improving SNN performance under ultra-low-latency conditions and is expected to promote practical neuromorphic hardware applications for further development.

**Performance:**

|  Dataset  |   Arch   |  ANN   |  T=1   |  T=2   |  T=4   |  T=8   |  T=16  |
| :-------: | :------: | :----: | :----: | :----: | :----: | :----: | :----: |
| CIFAR-10  |  VGG16   | 95.21% | 88.46% | 91.93% | 94.80% | 95.48% | 95.70% |
| CIFAR-10  | ResNet18 | 95.52% | 89.64% | 93.72% | 95.37% | 96.21% | 96.38% |
| CIFAR-10  | ResNet20 | 85.18% | 65.99% | 77.30% | 84.54% | 87.43% | 88.26% |
| CIFAR-100 |  VGG16   | 74.86% | 62.27% | 69.39% | 74.57% | 76.73% | 77.68% |
| CIFAR-100 | ResNet18 | 76.66% | 62.35% | 70.86% | 76.81% | 78.85% | 79.44% |
| CIFAR-100 | ResNet20 | 62.34% | 21.78% | 33.87% | 50.28% | 60.93% | 64.73% |

## Dependency

The major dependencies of this code are list as below.

```
torch==1.11.0+cu113
tqdm==4.65.0
numpy==1.25.2
torchvision==0.12.0+cu113
spikingjelly==0.0.0.0.1
```

## Environment

- System: Ubuntu 22.04.2 LTS (Jammy Jellyfish)
- GPU: NVIDIA GeForce RTX 3090
- CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz

## Usage

**Get info :**

```shell
python main.py --help
```

**If you want to train a NQ ANN model directly, you can consider executing the code like the following ways (for example: CIFAR100+VGG16)  :**

```shell
python main.py --dataset CIFAR100 --datadir {YOUR DATASET DIR} --net_arch vgg16 --batchsize 128 --L 4 --val_sim_len 4 --best_eval_snn_sim_len 4 --sim_len 64 --cutout_len 16 --lr 0.05 --CUDA_VISIBLE_DEVICES 1 --direct_training
```

**If you have already obtained a pretrained NQ ANN model, you can consider executing the code like the following ways (for example: CIFAR100+VGG16) :**

```shell
python main.py --dataset CIFAR100 --datadir {YOUR DATASET DIR} --load_model_name {YOUR SAVED MODELS DIR/MODEL NAME} --net_arch vgg16 --batchsize 128 --L 4 --sim_len 64 --cutout_len 16 --CUDA_VISIBLE_DEVICES 1
```

