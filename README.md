# dadapt_adan
Learning rate free learning for Adan[https://arxiv.org/abs/2208.06677], currently the most powerful optimizer. 
by D-Adaptation[https://arxiv.org/abs/2301.07733]
# Accuracy
All experiments use [ResNet18](https://arxiv.org/abs/1512.03385).
| Optimizer | Acc.        |epoch |
| ----------------- | ----------- | ----------- |
| [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)              | 93.02% | 200|
| [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)              | 93.02% | 200|
| [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)              | 93.02% | 200|
| [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)              | 93.02% | 200|
| [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)              | 93.02% | 200|

SGD is the experimental result of [kuangliu(pytorch-cifar)](https://github.com/kuangliu/pytorch-cifar).
# Acknowledgments
Many thanks to these excellent opensource projects
* [D-Adaptation](https://github.com/facebookresearch/dadaptation)
* [Adan](https://github.com/sail-sg/Adan)
* test code from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
