## PyTorch Code for 'E2-Train: Training State-of-the-art CNNs with Over 80% Less Energy'

## Introduction

PyTorch Implementation of our NeurIPS 2019 paper ["E2-Train: Training State-of-the-art CNNs with Over 80% Less Energy"](https://arxiv.org/abs/1910.13349).

This paper attempts to explore how to conduct more energy-efficient training of CNNs, so as to enable
on-device training? We strive to reduce the energy cost during training, by dropping
unnecessary computations, from three complementary levels: stochastic mini-batch
dropping on the data level; selective layer update on the model level; and sign
prediction for low-cost, low-precision back-propagation, on the algorithm level.

### PyTorch Model

- [x] ResNet
- [ ] MobileNet

## Dependencies

Python 3.7
* PyTorch 1.0.1
* CUDA 9.0
* numpy
* matplotlib


## Running E2-Train

* ResNet74

```bash
python main_all.py train cifar10_rnn_gate_74
```

## Citation

If you find this code useful, please cite the following paper:

    @article{E^2_train,
        title = {E2-Train: Training State-of-the-art CNNs with Over 80% Less Energy},
        author = {Wang, Yue and Jiang, Ziyu and Chen, Xiaohan and Xu, Pengfei and Zhao, Yang and Wang, Zhangyang and Lin, Yingyan},
        booktitle = {Advances in Neural Information Processing Systems 32},
        editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
        year = {2019}
        }

## Acknowledgment

We would like to thanks the arthor of [SkipNet](https://github.com/ucbdrive/skipnet). Our code implementation is highly inspired by this work.