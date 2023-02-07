# D-Adaptation Adan
Learning rate free learning for [Adan](https://arxiv.org/abs/2208.06677), currently the most powerful optimizer. 

by [D-Adaptation](https://arxiv.org/abs/2301.07733)
# Tips for Experiments
Experiments tips are based on [Adan](https://github.com/sail-sg/Adan) and [D-Adaptation](https://github.com/facebookresearch/dadaptation).
* Set the LR parameter to 1.0. This parameter is not ignored, rather, setting it larger to smaller will directly scale up or down the D-Adapted learning rate.
* Use the same learning rate scheduler you would normally use on the problem.
* It may be necessary to use larger weight decay than you would normally use, try a factor of 2 or 4 bigger if you see overfitting. D-Adaptation uses larger learning rates than people typically hand-choose, in some cases that requires more decay.
* Use the log_every setting to see the learning rate being used (d*lr) and the current D bound.
* The Adan IP variant implements a tighter D bound, which may help on some problems. The IP variants should be considered experimental.
* If you encounter divergence early on, and are not already using learning rate warmup, try change growth_rate to match a reasonable warmup schedule rate for your problem.
* D-Adapt Adan is relatively robust to `beta1`, `beta2,` and `beta3`, especially for `beta2`. If you want better performance, you can first tune `beta3` and then `beta1`.
* D-Adapt Adan's `weight_decay` recommends 0.02.
* Like Adan, we don't use a  restart strategy to make D-Adapt Adan simpler. But the restart strategy can further slightly improve the performance of D-Adapt Adan.
# Experiments results([cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html))
All experiments use [ResNet18](https://arxiv.org/abs/1512.03385).

Set to 50 epochs for quick experiments.
| Optimizer | Acc.        |epoch |
| ----------------- | ----------- | ----------- |
| [Adam](https://arxiv.org/abs/1412.6980)              | 92.77% | 50|
| [D-Adapt Adam](https://arxiv.org/abs/2301.07733)              | 89.99% | 50|
| [D-Adapt Adam IP](https://arxiv.org/abs/2301.07733)              | 90.81% | 50|
| [Adan](https://arxiv.org/abs/2208.06677)              | 92.7% | 50|
| D-Adapt Adan              | 92.64% | 50|
| D-Adapt Adan IP             | 93.04% | 50|


![fig](https://user-images.githubusercontent.com/64115820/217195448-7202126f-6682-4fb0-9c99-432f534a9c9c.png)

# Run experiment
```
git clone https://github.com/qwopqwop200/dadapt_adan.git
cd dadapt_adan/test
python main.py --opt d-adan #[adam,adan,d-adam,d-adam-ip,d-adan,d-adan-ip]
```

# Pseudo code
<img width="485" alt="sudo" src="https://user-images.githubusercontent.com/64115820/217242205-efcb5d6e-9123-4ce4-bf31-3ffcefb002b2.png">
where λ is the weight decay constant.

# Acknowledgments
Many thanks to these excellent opensource projects
* [D-Adaptation](https://github.com/facebookresearch/dadaptation)
* [Adan](https://github.com/sail-sg/Adan)
* test code from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
