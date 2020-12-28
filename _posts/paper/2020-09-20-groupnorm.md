---
layout: post
title: "Group Normalization"
description: "Yuxin Wu, Kaiming He (ECCV 2018)"
date: 2020-09-20
categories: paper
tags: [review, deeplearning, normalization]
image: 
---

# [1] 아이디어 제안 배경

여러 Normalization 방법론이 등장했습니다. Batch Normalization을 시작으로, Recurrent 모델에 적용가능한 형태인 Layer Normalization, 그 밖에는 Weight Normalization, Instance Normalization 등이 존재합니다. 일반적으로 Batch Normalization이 이미지 처리 분야에서 범용적으로 사용되는데, 배치 단위로 평균과 분산을 구하기 때문에 배치의 크기가 줄어들수록 성능이 낮아지는 단점이 있었습니다.


배치 크기는 하드웨어 성능이 점점 좋아지면서 걱정할 필요가 없어질 것이라 생각할 수도 있습니다. 그러나 대표적인 이미지 처리 분야인 Object Detection이나 Segmentation에서는 입력으로 들어가는 이미지의 해상도가 매우 높아 배치 크기를 1 또는 2 정도밖에 설정하지 못한다고 합니다(예: Fast/er R-CNN). 이런 상황에서 Batch Normalization의 효과는 오히려 성능을 떨어뜨리는 원인이 됩니다. 이에 본 논문에서는 배치에 대한 의존성을 없애면서 낮은 배치에서도 준수한 성능을 낼 수 있는 방법을 제시합니다. 미리 결과를 엿보자면 아래와 같습니다.

<img src="/assets/figures/gn_result.PNG" width="70%">


# [2] 방법론

## Overview
사실 Group Normalization을 비롯하여 지금까지 제시된 방법론은 아래의 그림으로 깔끔하게 정리할 수 있습니다. 직관적인 그림이라 누구나 쉽게 컨셉을 이해하실 수 있을 것 같습니다.

<img src="/assets/figures/gn_overview.PNG" width="70%">

그림을 통해 직관적으로 이해하자면 Group Normalization은 Layer Normalization과 Instance Normalization을 절충한 형태로 볼 수 있겠습니다. [저자의 발표자료](http://kaiminghe.com/eccv18gn/group_norm_yuxinwu.pdf)를 인터넷에서 접할 수 있어 그 중 하나의 슬라이드를 아래에 첨부합니다. 각 Normalization 방법론을 간단하게 비교하였습니다.

<img src="/assets/figures/gn_compare.PNG" width="70%">

## Group Normalization

SIFT, HOG와 같이 관련있을 법한 특징을 그룹짓는 방법론은 이전에도 존재했다고 합니다(두 방법론에 대한 소개는 참고자료에 첨부하였습니다.). Group Normalization 역시 이름에 맞게 feature를 그룹으로 나누어 각각 normalize하는 방법을 택하였습니다. 배치가 아닌 feature를 단위로 그룹짓기 때문에 배치의 크기에 구애받지 않고 적용할 수 있다는 것이 장점입니다. 따라서 sequential 또는 generative 모델에도 적용할 수 있다고는 하는데, 이 부분에 대해서는 future work로 남겨두었습니다.


2차원 이미지에 대한 feature를 $(N, C, H, W)$의 벡터로 표현할 수 있습니다. 각각 배치 크기, 채널, 높이, 너비를 의미합니다. 그리고 각 Normalization 방법론에 대한 수식을 비교하면 아래와 같습니다. Overview의 그림과 함께 보시면 이해가 쉽습니다.

- Batch Normalization <br>
  
$$ S_i=\left\{ k|k_C=i_C \right\} $$

- Layer Normalization <br>
  
$$ S_i=\left\{ k|k_N=i_N \right\} $$

- Instance Normalization <br>
  
$$ S_i=\left\{ k|k_N=i_N, k_C=i_C \right\} $$

- Group Normalization <br>
  
$$ S_i=\left\{ k|k_N=i_N, \lfloor{k_C \over C/G}\rfloor = \lfloor{i_C \over C/G}\rfloor \right\} $$


$\lfloor{k_C \over C/G}\rfloor = \lfloor{i_C \over C/G}\rfloor$는 $i$와 $k$가 같은 그룹의 채널에 속한다는 것을 의미합니다. [Floor function](https://ko.wikipedia.org/wiki/%EB%B0%94%EB%8B%A5_%ED%95%A8%EC%88%98%EC%99%80_%EC%B2%9C%EC%9E%A5_%ED%95%A8%EC%88%98)에 대해 알아보시면 무슨 뜻인지 쉽게 이해할 수 있습니다.


논문에서는 친절하게 TensorFlow로 구현한 코드까지 알려줍니다. 간단한 코드기 때문에 아래에 옮겨두었습니다.

```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: learnable scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN

    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])

    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x − mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [N, C, H, W])

    return x * gamma + beta
```


# [3] 실험
Group Normalization의 주 목적 중 하나는 배치 크기에 영향을 많이 받는 Batch Normalization의 단점을 극복하는 것입니다. 그렇기 때문에 비교를 위한 실험의 베이스라인을 ResNet + BN으로 설정합니다.


실험에 대한 결과는 카테고리를 몇 개 나누어 보여드리도록 하겠습니다.

### 같은 배치 크기에서의 방법론 간 비교 (BN/LN/IN/GN)

<img src="/assets/figures/gn_cate1.PNG" width="70%">

<img src="/assets/figures/gn_cate2.PNG" width="70%">

배치 크기가 클 땐 역시 BN이 좋은 성능을 보이지만, 이전의 LN, IN보다는 GN이 준수한 성능을 내는 것을 확인할 수 있습니다. 또한 train 데이터셋에 대해서는 GN이 BN보다 낮은 에러율을 보이고 있습니다. 이는 GN의 최적화 효과가 좋음을 의미하며, val 데이터셋에 대해 에러율이 BN보다 높은 것은 BN에 비해 regularization 효과는 조금 떨어진다는 것을 의미합니다. 논문에서는 BN의 배치 샘플 간 normalization 적용이 uncertainty를 더 증가시키기 때문이라 말하며, 적절한 regularizaer와 GN을 혼합하면 더 좋은 성능을 낼 것이라 기대합니다. 이는 future work로 남겨두었습니다.


### 배치 크기에 따른 비교 (BN vs. GN)

<img src="/assets/figures/gn_cate3.PNG" width="70%">


<img src="/assets/figures/gn_batch.PNG" width="70%">

배치 크기가 8로 줄어들 때부터 GN의 성능이 더 높아짐을 확인할 수 있습니다.



### 그룹 수에 따른 비교 (GN)

<img src="/assets/figures/gn_cate4.PNG" width="70%">

논문의 실험 셋팅에서는 그룹의 수가 32개일 때, 그룹 당 채널의 수가 16개일 때 가장 좋은 성능을 보였습니다. 참고로 그룹의 수가 1개인 것은 LN과 동일합니다.

# [4] 마치며
본 논문에서는 기하학적으로 방법론의 효과를 증명한 LN과는 약간 다른 방향, 즉 많은 실험을 통해 성능을 검증하였습니다. 논문에는 깊은 모델에 대한 실험, Batch Renormalization과의 비교, Detection/Segmentation 모델에 대한 실험을 포함하여 몇 가지 실험이 더 있어서 필요하신 분은 참고하시면 좋을 것 같습니다.


# [5] 참고자료

- [[Paper] Group Normalization](https://arxiv.org/abs/1803.08494)
- [[Presentation] Group Normalization](http://kaiminghe.com/eccv18gn/group_norm_yuxinwu.pdf)
- [[Presentation] 특징 기술자(Feature Descriptor)](http://166.104.231.121/ysmoon/mip2017/lecture_note/%EC%A0%9C6%EC%9E%A5-%EC%B6%94%EA%B0%80.pdf)