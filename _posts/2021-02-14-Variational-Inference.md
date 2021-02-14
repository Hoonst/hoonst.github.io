---
layout: post
title: "Variational Inference"
description: "Basic but complicated Theory"
tags: [paper_review]
date: 2021-02-11
comments: true
typora-root-url: ../../hoonst.github.io
---

# Introduction

 $p(x)$는 데이터 $x$의 Representation distribution을 찾기 위함

$p(y|x)$: Conditional Model, $x$ 분포는 딱히 상관없다.



Latent Variable Models: Probabilistic Model 예시 중 하나

$p(x) / p(y|x)$의 표기가 있을 때, $y$ 는 Query라고 부르고 $x$는 Evidence라고 부른다.

$p(x) = \sum\limits_x p(x|z) p(z)$

$p(y|x) = \sum\limits_z p(y|x,z)p(z)$ 

$p(x)$는 일반적으로 복잡한 분포인데, $p(z)$는 꽤 간단한 분포로 설정함 > Gaussian



$p(x|z) = N(\mu(z), \sigma(z))$ 파라미터를 구하는 과정은 복잡할 수 있어도, 해당 분포도 매우 간단



$p(x) = \int p(x|z)p(z)dz$

두 분포는 쉽게 구할 수 있지만, Product는 간단하지 않을 수 있다.



Latent Variable Model을 어떻게 훈련시키는가?

$p_\theta(x) = \int p(x|z)p(z)dz$

$\theta \leftarrow arg max_\theta \frac{1}{N} \sum\limits_i log(\int p_\theta(x_i |z)p(z)dz)$

$\int p_\theta(x_i |z)p(z)dz$ 이 적분식이 Intractable

애초에 MLE는 Optimization으로 어떻게 구하지?



tractable하게는 expected log-likelihood

$\theta \leftarrow arg max_\theta \frac{1}{N} \sum\limits_i E_{z~p(z|x_i)}[log p_\theta(x_i, z)]$

guess the latent variable를 진행함에 있어 무수히 후보가 많을 것이므로, $p(z|x_i)$를 하나의 분포로 설정



Lower Bound를 생성하고 그 Lower Bound를 최대화 하는 것이 필요하다.



구하고자 하는 $log p(x_i)$의 Lower Bound를 Latent Variable Z로 구하자