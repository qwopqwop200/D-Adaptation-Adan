# D-Adaptation Adan
Learning rate free learning for [Adan](https://arxiv.org/abs/2208.06677), currently the most powerful optimizer. 

by [D-Adaptation](https://arxiv.org/abs/2301.07733)
# Tips for Experiments
Experiments tips are based on [Adan](https://github.com/sail-sg/Adan) and [D-Adaptation](https://github.com/facebookresearch/dadaptation).
* Set the LR parameter to 1.0. This parameter is not ignored, rather, setting it larger to smaller will directly scale up or down the D-Adapted learning rate.
* Use the same learning rate scheduler you would normally use on the problem.
* It may be necessary to use larger weight decay than you would normally use, try a factor of 2 or 4 bigger if you see overfitting. D-Adaptation uses larger learning rates than people typically hand-choose, in some cases that requires more decay.
* Use the log_every setting to see the learning rate being used (d*lr) and the current D bound.
* The D-Adaptation Adan IP variant implements a tighter D bound, which may help on some problems. The IP variants should be considered experimental.
* If you encounter divergence early on, and are not already using learning rate warmup, try change growth_rate to match a reasonable warmup schedule rate for your problem.
* D-Adaptation Adan is relatively robust to `beta1`, `beta2,` and `beta3`, especially for `beta2`. If you want better performance, you can first tune `beta3` and then `beta1`.
* D-Adaptation Adan's `weight_decay` recommends 0.02.
* Unlike Adan, D-Adaptation Adan appears to have little or no performance benefit from the restart strategy. For this reason, it is recommended not to use the restart strategy.
# Experiments results([cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html))
All experiments use [ResNet18](https://arxiv.org/abs/1512.03385).

Set to 50 epochs for quick experiments.

| Optimizer | Acc.        |epoch |
| ----------------- | ----------- | ----------- |
| [Adam](https://arxiv.org/abs/1412.6980)              | 92.77% | 50|
| [D-Adaptation Adam](https://arxiv.org/abs/2301.07733)              | 89.99% | 50|
| [D-Adaptation Adam IP](https://arxiv.org/abs/2301.07733)              | 90.81% | 50|
| [Adan](https://arxiv.org/abs/2208.06677)              | 92.7% | 50|
| D-Adaptation Adan              | 92.64% | 50|
| D-Adaptation Adan IP             | 93.04% | 50|
| [Adan with restart](https://arxiv.org/abs/2208.06677)              | 93.31% | 50|
| D-Adaptation Adan with restart             | 92.8% | 50|
| D-Adaptation Adan IP with restart           | 92.94% | 50|

![fig](https://user-images.githubusercontent.com/64115820/217195448-7202126f-6682-4fb0-9c99-432f534a9c9c.png)

# Run experiment
```
git clone https://github.com/qwopqwop200/dadapt_adan.git
cd dadapt_adan/test
python main.py --opt d-adan #[adam,adan,d-adam,d-adam-ip,d-adan,d-adan-ip]
#If you want restart strategy
python main.py --opt d-adan --restart #[adan,d-adan,d-adan-ip]
```
# Implementation of restart strategy
The restart strategy gets better performance by resetting the momentum term every N steps.

Adan with restart strategy on cifar10 has a performance advantage.(92.7 -> 93.31)

If you simply reset the momentum term, such as Adan in D-Adaptation Adan, the model diverges.To prevent this, we reset the s and gsq_weighted(or numerator_weighted).

However, there is little or no performance benefit in these cases.

Also, reset D together causes the model to fall into a local minimum, which is why we don't reset D.

The reason the model diverges is because of gsq_weighted(or numerator_weighted).

If only gsq_weighted(or numerator_weighted) is reset, there is little performance difference from reset with s and gsq_weighted(or numerator_weighted).

And it doesn't seem like a good choice mathematically.

So I implemented s and gsq_weighted(or numerator_weighted) to be reset together.

As said above, these implementations appear to offer little or no performance benefit.

# Pseudo code
<img width="485" alt="sudo" src="https://user-images.githubusercontent.com/64115820/217242205-efcb5d6e-9123-4ce4-bf31-3ffcefb002b2.png">
where λ is the weight decay constant.

# Acknowledgments
Many thanks to these excellent opensource projects
* [D-Adaptation](https://github.com/facebookresearch/dadaptation)
* [Adan](https://github.com/sail-sg/Adan)
* test code from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
