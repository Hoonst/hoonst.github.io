---
layout: post
title: "Information Theory"
description: "Basic but complicated Theory"
tags: [paper_review]
date: 2021-02-11
comments: true
typora-root-url: ../../hoonst.github.io
---

# Introduction

Entropy: 정보를 표현하는데 있어서 필요한 최소 평균 자원량

정보를 어떻게 표현할 수 있을까? Common Currency = bits

Shannon



연인의 카톡에서 하트와 'ㅗ'의 빈도를 살펴보면 당연히 하트가 훨씬 더 많을테이고, 하트의 bit 수를 크게 설정하면 전체 정보를 표현하기 위한 자원량이 커질 것이므로, 작게 표현해야 합니다. 즉, 사건의 확률이 클수록 bit 수를 작게 표현해야 합니다.

예를 들어, 하트를 111, 'ㅗ'를 0으로 표현했을 때의 전체 메시지의 필요 자원과, 반대의 경우는 당연히 후자가 더 자원을 아낄 수 있습니다. 



모든 정보에 대한 가정은 Random 하기에 확률로 표현할 수 있는 것이 새넌의 가정

Entropy 식은 $\sum\limits_{i}-p_i log p_i$으로 표현할 수 있으며, 아무리 코딩을 잘해도 이거보다 더 낮은 값을 가질 수 없는 Lower Bound의 역할을 합니다. 

즉, 어떤 사건의 등장 확률이 크다면 표현하는 길이가 작아져야 하고, 반대라면 커져야 하는 것이 섀넌의 주장이며 Log로 나타내는 것입니다. 그런데 이 최소 자원량을 Random하게 나타내야 한다면 기대값으로 표현해야 합니다. 그리고 기대값이 정보의 최소 평균 길이임을 섀넌이 밝혔다고 합니다. 

여기서 또 중요한 점은 '효율적인 코딩'을 하기 위해선 사건의 등장 확률이 개별적으로 달라야 하며, 이를 통해 확률분포가 들쑥날쑥하게 변해야 합니다. 즉, Uniform하면 안된다는 것이죠. 따라서 확률분포가 Uniform일 경우 Lower Bound의 최대값을 갖게 됩니다. 이는 모든 사건이 동일하게 표현했다는 뜻이  됩니다. 

Continuous할 때는 Gaussian일때 최대가 된다고 합니다. 



Cross Entropy

Cross Entropy는 실제 Entropy와 자신이 가정한 확률 분포 상의 차이라고 생각하시면 됩니다. Entropy가 최소 평균 자원량인데, 어떠한 이유로 가정을 다르게 줘서, 예를 들어, 카톡의 예시에서 하트와 'ㅗ'의 확률을 각기 다르게 설정하지 않고 Uniform하게 설정하게 된다면 차이가 발생하게 될 것입니다. 

$\sum\limits_i (p_i -log_2 q_i)$로 나타내는데 이 때 $q_i$가 '내가 생각한 멍청한 가정'이라고 할 수 있습니다. 



KL Divergence

KL Divergence는 내가 등신같이 짠 코딩보다 실제 코딩과의 차이를 나타냅니다. 따라서, 확률 분포의 차이의 의미를 갖게 되는 것입니다. Divergence는 양수의 값을 가져야하므로, '내가 등신 같이 생각한 가정'이 더 값이 크므로 이 값에서 Entropy값을 뺀 것입니다. 

KL Divergence = $\sum\limits_i (-p_ilog_2 q_i) - \sum\limits_{i}-p_i log p_i = \sum\limits_ip_ilog_2\frac {p_i}{q_i}$

이는 즉, 내가 얼마나 등신같이 생각했는가에 대한 차이 값이라고 할 수 있습니다. 



Mutual Information

$\sum_i\sum_jp(x_i, y_i) log \frac {p(x_i, y_i)}{p(x_i)p(y_i)}$

그 꼴을 그대로 가져오면, 원래는 $p(x_i, y_i)$인데, 즉 실제로는 독립이 아닐 때, 내가 두 확률분포가 독립이다 라고 생각했을 때의 차이 값

실제로 $p(x_i, y_i)$이 독립이라면, Log에 해당하는 값은 0이 되고 Mutual Information 값이 0이 됩니다. 즉 독립일 때의 차이가 얼마나 떨어져 있느냐, 즉 얼마나 서로 종속적이냐를 알 수 있는 지표로서 활용할 수 있게 됩니다. 

