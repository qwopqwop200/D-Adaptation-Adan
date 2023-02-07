# D-Adaptation Adan
Learning rate free learning for [Adan](https://arxiv.org/abs/2208.06677), currently the most powerful optimizer. 

by [D-Adaptation](https://arxiv.org/abs/2301.07733)
# Experimental results([cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html))
All experiments use [ResNet18](https://arxiv.org/abs/1512.03385).

Set to 50 epochs for quick experiment.
| Optimizer | Acc.        |epoch |
| ----------------- | ----------- | ----------- |
| [Adam](https://arxiv.org/abs/1412.6980)              | 92.77% | 50|
| [D-Adapt Adam](https://arxiv.org/abs/2301.07733)              | 89.99% | 50|
| [D-Adapt Adam IP](https://arxiv.org/abs/2301.07733)              | 90.81% | 50|
| [Adan](https://arxiv.org/abs/2208.06677)              | 92.7% | 50|
| D-Adapt Adan              | 92.64% | 50|
| D-Adapt Adan IP             | 93.04% | 50|
# Acknowledgments
Many thanks to these excellent opensource projects
* [D-Adaptation](https://github.com/facebookresearch/dadaptation)
* [Adan](https://github.com/sail-sg/Adan)
* test code from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
