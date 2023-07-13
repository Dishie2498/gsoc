---
title: Bootstrap
layout: post
post-image: "https://i.stack.imgur.com/f6pXo.png"
description: Implementation of the boostraps for selecting significant multiplets + confidence interval estimation.
tags:
- bootsrap
- confidence intervals
- resampling
---
### - Resampling the Input Data
To perform bootstrapping, we need to resample the input data. This involves randomly sampling data points from the 
original dataset with replacement. Each resampled dataset should have the same length as the original dataset and 
values can be repeated within.

### - Selecting Significant Multiplets 
Once we have resampled datasets, we can compute the desired metric, o-information on each resampled dataset. This 
step allows us to evaluate the metric's value on different subsets of the original data.

### - Compute Confidence Intervals
With the resampled metric values, we can estimate confidence intervals to quantify the uncertainty associated with 
the metric. One common approach is to calculate the percentiles of the resampled metric values. For example, computing 
the 5th and 95th percentiles will give a 90% confidence interval. These percentiles represent the lower and upper 
bounds of the confidence interval, respectively.
