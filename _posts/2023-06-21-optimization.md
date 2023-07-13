---
title: Optimization of Code
layout: post
post-image: "https://m.media-amazon.com/images/I/41OMvm3ucaL._SX300_SY300_QL70_FMwebp_.jpg"
description: Reducing coputation times in calculating o-information.
tags:
- sample
- post
- test
---
The information-theoretical quantity known as the O-information (short for "information about Organisational structure") is used 
to characterise statistical interdependencies within multiplets of three and more variables. It enables us to determine the nature of 
the information, i.e., whether multiplets are primarily carrying redundant or synergistic information, in addition to quantifying how 
much information multiplets of brain areas are carrying.
It takes an extensive amount of computation to estimate HOIs. The O-information is a perfect choice to estimate HOIs in a timely manner
because its computational cost just requires basic quantities like entropies. There is yet no neuroinformatic standard of merit for HOI 
estimation that can be used by aficionados of all skill levels in a reasonable amount of time.

## [jax.vmap()](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)
[```vmap```](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) is a feature in JAX that enables efficient parallelization of functions over arrays or sequences of inputs.
- vmap stands for "vectorized map" and is a powerful feature that enables efficient parallelization of functions over arrays or sequences of inputs.
- With vmap, you can seamlessly apply a function to a batch of inputs, eliminating the need for explicit looping. This not only simplifies the code but also significantly improves performance, especially when dealing with large datasets or complex computations.

### Following is the comparison in computation times between tensor and vmap implementation 
[comparison graph](https://drive.google.com/uc?id=1Y0mtv3flyzhtfjgR3P6UXl499FUUFqyA)
<a href="https://drive.google.com/uc?export=view&id=1Y0mtv3flyzhtfjgR3P6UXl499FUUFqyA"><img src="https://drive.google.com/uc?export=view&id=1Y0mtv3flyzhtfjgR3P6UXl499FUUFqyA" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture" />

### Following PRs implement vmap in calculating o-information
- [Pull request #6](https://github.com/brainets/hoi/pull/6)
- [Pull request #7](https://github.com/brainets/hoi/pull/7)
