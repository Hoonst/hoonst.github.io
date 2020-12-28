---
layout: post
title: "Layer Normalization"
description: "Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (NIPS 2016)"
date: 2020-09-18
categories: paper
tags: [review, deeplearning, normalization]
image: 
---


# [1] 아이디어 제안 배경
이전 [Batch Normalization 관련 포스트](https://youngerous.github.io/paper/2020/09/03/batchnorm/) 마지막 부분에서, 해당 방법론의 Future Work로 Recurrent 기반의 모델에 적용하는 발전 방향을 제시하였다고 이야기하였습니다. Layer Normalization은 Batch Normalization의 방법론과 유사하지만 간단한 변형으로 Recurrent 기반 모델에서도 잘 동작하는 방법론입니다.


간단하게 복습하자면 Batch Normalization은 신경망의 각 레이어마다 입력값의 분포가 달라지는 covariate shift 현상을 없애기 위해 제안되었습니다. 각 비선형 활성화 함수에 들어가기 전의 값을 normalize하는 간단한 방법으로 문제를 해결하려 하였습니다. 또한 normalize를 위해 전체 데이터셋에 대한 분산을 구하는 비효율을 극복하기 위해 mini-batch 단위로 범위를 제한하여 평균과 분산을 추정하였습니다.


## Batch Normalization의 문제점

### 문제점 1: Mini-batch의 크기에 의존적
그러나 기존 Batch Normalization 방법론의 문제점이 존재합니다. 먼저, 해당 방법론의 효과는 mini-batch의 크기에 의존적입니다. 충분히 큰 mini-batch에 대해서는 상관이 없겠지만, 만약 그 크기가 작다면 학습에 문제를 일으킬 수 있습니다. 극단적으로 mini-batch의 크기가 1이면 당장 분산이 0이 되어버리기 때문에 학습에 적용할 수 없습니다(논문의 표현을 빌리면 *pure online regime*에서 사용할 수 없습니다). Mini-batch의 크기가 작아질 수밖에 없는 분산(*extremely large distributed*) 모델에서도 역시 BN을 적용하는 것이 어렵습니다.


### 문제점 2: Recurrent 기반 모델에 적용 어려움

Feed-forward 형태의 신경망에서는 전파 과정 사이에 BN Layer를 단순히 삽입하는 형태입니다. 조금 더 구체적으로는 배치 단위 입력값의 평균 통계량(평균, 표준편차)을 계산하여 normalize하는 형태입니다. 각 배치 별 입력값의 형태가 동일하기 때문에 간편하게 적용할 수 있습니다.


잠깐 recurrent 기반 모델의 특징에 대해 짚어보겠습니다. Recurrent 기반의 모델의 입력값은 일반적으로 (길이가 서로 다른)sequence입니다. 그리고 모든 time-step에서 동일한 가중치(weight)를 공유하는 특징이 있습니다. 이러한 형태를 갖는 모델에서 Batch Normalization을 적용하기 위해서는 매 time-step마다 별도의 통계량을 저장해야 합니다. 이는 모델을 훨씬 복잡하게 만드는 일입니다. 특히 평가를 위한 sequence가 학습에서 사용된 모든 sequence의 길이보다 긴 경우 문제가 됩니다.


# [2] 방법론

## Weight Normalization

사실 Layer Normalization 이전에 [Weight Normalization](https://arxiv.org/abs/1602.07868)이라는 기법도 등장했습니다. 간단하게만 언급하자면, mini-batch의 입력값이 아닌 **레이어의 가중치(weight)** 를 normalize하는 기법입니다. 또한 *mean-only batch normalization*이라는 변형된 normalizing 방법을 함께 제시합니다. 일반적으로 normalize를 하기 위해서는 값에 평균을 뺀 후 표준편차로 나누는 과정을 수행합니다. 그러나 여기서는 표준편차로 나누는 과정을 빼고 평균만 빼는 방법을 제안합니다. 


이 방법이 연산 측면에서도 더 효율적이고(일반적으로 CNN 모델의 가중치가 입력 feature보다 적은 차원을 갖는 것 또한 연산 효율에 도움이 됨), 노이즈 자체가 완만하게 생성된다고 합니다.


## Layer Normalization

### Intuition
논문의 표현으로 Batch Normalization과 Layer Normalization의 차이를 설명하면 아래와 같습니다.

> Batch Normalization: Estimate the normalization statistics from the summed inputs to the neurons **over a mini-batch of training case** 


> Layer Normalization: Estimate the normalization statistics from the summed inputs to the neurons **within a hidden layer**


아래의 그림은 Layer Normalization을 가장 직관적으로 나타내는 그림입니다. 기본적으로 Batch Normalization처럼 값을 normalize하지만, batch 단위가 아닌 input을 기준으로 평균과 분산을 계산합니다.


<img src="https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/01/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-01-11-11.48.12.png?resize=1024%2C598" width="70%">


두 방법론의 차이를 수식으로 비교해보겠습니다. 먼저 Batch Normalization에서 평균과 분산은 다음과 같이 계산합니다. Batch size가 $n$인 입력값에 대해 첫 번째($j=1$) 값끼리, 두 번째($j=2$) 값끼리 계산하는 방식으로 수행됩니다. 위의 그림에서 $n$은 3, $j$의 최댓값은 6이 되겠습니다.


$$ \mu_j = {1\over n}\sum_{i=1}^n x_{ij}\, , \; \;\; \sigma_j^2= {1\over n}\sum_{i=1}^n (x_{ij}-\mu_j)^2$$


반면 Layer Normalization의 계산 수식은 다음과 같습니다. 하나의 sample $i$의 모든 hidden unit($m$개)에 대한 평균과 분산을 계산하는 방식입니다. 위의 예시에서 $m$은 6, $i$의 최댓값은 3이 되겠습니다.


$$ \mu_i = {1\over m}\sum_{j=1}^m x_{ij}\, , \; \;\; \sigma_i^2= {1\over m}\sum_{j=1}^m (x_{ij}-\mu_i)^2$$

### 특징
위와 같은 방법, 즉 mini-batch 단위가 아닌 각 sample의 hidden unit 단위로 normalization을 수행하는 것은 아래와 같은 특징을 갖습니다.

1. 데이터마다 각각 다른 normalization term($\mu$, $\sigma$)을 갖습니다.
2. Mini-batch의 크기에 영향을 받지 않습니다(batch size가 1인 경우에도 동작합니다).
3. 서로 다른 길이를 갖는 sequence가 batch 단위의 입력으로 들어오는 경우에도 적용할 수 있습니다(1번의 특징 때문입니다).


일반적인 RNN 모델에서는 많은 time-step을 거칠 때 gradient가 explode 또는 vanish하는 현상이 발생합니다. 그러나 Layer Normalization이 적용된다면 레이어에 입력으로 들어오는 값의 scale에 영향을 받지 않기 때문에 더 안정적인 결과를 보인다고 합니다.

### Invariance Properties

본 논문에서는 Batch Normalization, Weight Normalization, 그리고 제안하는 방법론인 Layer Normalization이 갖는 invariance 특성을 비교합니다. 이를 표로 나타내면 아래와 같습니다.


<img src="/assets/figures/ln_table.PNG" width="70%">


### 참고: Layer Normalization의 효과에 대한 증명
본 논문에서는 기하학적인 관점에서 학습이 어떻게 일어나고, Normalization을 하게 되면 gradient가 어떤 과정으로 안정화되는지 심도있게 다루었습니다. 구체적으로는 Normalization Scalar $\sigma$가 **implicitly하게 learning rate를 감소시키고 학습을 안정적으로 진행시키는지**를 증명하였습니다. 덕분에 따라가는 데 정말 많은 시간이 들었지만, 포스팅에서는 최대한 간결하고 직관적으로 전달해보도록 하겠습니다. 필요하신 분만 참고하시면 좋을 것 같습니다.


**1) Riemannian Metric** <br>
먼저 파라미터 공간을 기하학적인 관점으로 살펴보겠습니다. 일반적으로 학습 가능한 파라미터로 구성된 통계적 모델은 Smooth Manifold(미적분학을 전개할 수 있는 구조가 주어진 manifold; Differentiable Manifold)를 생성합니다. 그리고 모델의 출력값이 확률분포를 갖는다면, manifold 위에 있는 두 지점의 분리 정도를 측정하는 대표적인 방법은 KL Divergence가 있습니다.


KL Divergence Metric 가정 하에 파라미터 공간은 Riemannian Manifold로 정의할 수 있습니다. Riemannian Manifold를 한 마디로 표현하자면 두 점 사이의 거리를 측정할 수 있는 Smooth Manifold라 할 수 있습니다. 우리는 gradient가 안정화되는 과정을 보고 싶기 때문에 이 Riemannian Manifold의 곡률(curvature)을 알아야 합니다.


Riemannian Manifold의 곡률은 Riemannian Metric을 통해 확인할 수 있습니다. $ds^2$의 이차형식으로 표현가능한 Riemannian Metric은 파라미터 공간의 한 지점의 tangent space에서의 매우 작은 거리($\delta$)입니다. 한 지점에서 tangent space를 구성하는 축인 법선 벡터, 접선 벡터, binomial 벡터 정보를 알게 되면 공간이 어떤 식으로 기울어져 있는지를 확인할 수 있습니다. 정리하자면 Riemannian Metric을 통해서 Riemannian Manifold의 기울어진 정도(곡률)를 확인할 수 있다는 뜻입니다.


그런데 KL Divergence 형태의 Riemannian Metric은 Fisher Information Matrix의 2차 테일러 전개에 의해 잘 근사할 수 있다는 것이 선행연구를 통해 증명되었다고 합니다. Fisher Information Matrix는 어떤 확률변수의 관측값으로부터 파라미터에 대해 유추할 수 있는 정보의 양을 의미합니다. 이는 식으로 나타내면 log likelihood의 미분값을 두 번 곱한 값의 기댓값이고, [참고한 글](https://www.facebook.com/buckeyestatfisher/posts/342873913320663)에서 언급한 직관적인 의미는 아래와 같습니다. 아래의 항목은 모두 같은 의미를 지니고, 반대의 경우에 대해서는 뒤집어서 생각하시면 됩니다.

   - Fisher Information이 (양의 방향으로) 매우 크다.
   - log likelihood의 이계도함수 값이 (음의 방향으로) 매우 작다.
   - log likelihood가 봉우리 근처에서 매우 급격하게 변화한다.
   - log likelihood의 봉우리가 매우 뾰족하다.
   - 봉우리를 정확하게 찾기가 매우 쉬워진다.
   - 파라미터에 대한 정확한 정보를 얻을 수 있다.


Fisher Information Matrix를 이용하여 Riemannian Metric을 근사한 식은 아래와 같습니다. $\delta$는 파라미터에 주어지는 아주 작은 변화량입니다.

<img src="/assets/figures/ln_kl.PNG" width="70%">


**2) The geometry of normalized generalized linear model** <br>
앞서 확인한 파라미터 공간의 기하학적인 관점을 유지한 채, Generalized Linear Model(GLM)에서 Normalization이 어떻게 작용하는 지 알아보겠습니다. GLM은 종속변수가 정규분포, 이항분포, 포아송분포 등 다양한 분포를 가질 수 있습니다. 


Multi-dimensional GLM 모델의 각 파라미터에 대한 Fisher Information Matrix는 아래와 같이 나타냅니다. $\otimes$는 일반화된 outer product 형태인 [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product)을 의미합니다. 

<img src="/assets/figures/ln_fisher.PNG" width="70%">


Kronecker product의 결과값은 block matrix의 형태를 따르는데요, 여기에 Normalization을 적용하면 아래와 같이 표현할 수 있습니다. 즉 $\bar{F}$는 각 파라미터에 대해 유추할 수 있는 정보의 양인 것입니다.

<img src="/assets/figures/ln_kronecker.PNG" width="70%">


위 그림을 직관적으로 해석하자면 다음과 같습니다. Normalization이 적용된 상황에서 가중치 벡터 $w_i$를 스케일링하면 모델의 출력값은 그대로입니다. 그러나 Normalization term($\sigma$)에 의해 Fisher Information Matrix으로 표현되는 $w_i$ 방향에서의 곡률은 그에 반비례하게 줄어들게 됩니다. 파라미터를 업데이트하는 방법은 같지만 Normalization이 적용되었을 때 learning rate를 더 효과적으로 컨트롤할 수 있게 되었다고도 볼 수 있습니다.


또한 일반적인 GLM 모델에서 $g$로 표현되는 Gain 파라미터가 입력값의 norm에 영향을 받는 반면, Normalization이 적용된 후에는 prediction error에만 영향을 받는 것을 확인할 수 있습니다. 이는 입력값의 scaling에 더 강건하다(robust)는 것을 의미합니다.



# [3] 실험
실험은 아래의 여섯 가지 task에 대해 진행되었습니다. Recurrent 기반의 task에 대해 조금 더 초점을 맞추었다고 합니다.

1. Image-Sentence Ranking
2. Question-Answering
3. Contextual Language Modeling
4. Generative Modeling
5. Handwriting Sequence Generation
6. MNIST Classification


LSTM 모델을 예로 들어 LN을 적용한 식을 어떻게 표현하는 지만 알아보겠습니다. 일반적인 LSTM을 식으로 나타내면 아래와 같습니다.

<img src="/assets/figures/ln_lstm1.PNG" width="70%">

그리고 LN을 적용하면 아래와 같이 나타낼 수 있습니다. $\alpha$와 $\beta$는 각각 additive/multiplicative 파라미터입니다.

<img src="/assets/figures/ln_lstm2.PNG" width="70%">


각 실험에 대한 내용은 논문에 잘 나와 있기 때문에 위의 방법론이 실제로 잘 적용되는지에 대한 결과 그림 몇 장을 첨부하는 것으로 대신하겠습니다. 


<img src="/assets/figures/ln_exp1.PNG" width="70%">


<img src="/assets/figures/ln_exp2.PNG" width="70%">


다만 Convolutional Networks에서는 LN을 적용하는 것이 속도 측면에서는 이점이 되지만, 성능 자체는 BN이 더 좋다고 언급합니다. Fully-connected layer의 은닉 노드는 출력값을 만드는 데 기여하는 정도가 각각 비슷하기 때문에 입력값(summed inputs)을 scaling하거나 centering하는 것이 효과가 있지만, CNN에서는 그렇지 않습니다. 이미지의 가장자리 부분을 receptive field로 하는 은닉 노드는 거의 활성화되지 않아 다른 부분의 은닉 노드와는 매우 다른 통계량을 갖습니다. 즉 feature 단위로 normalization을 할 경우 이미지의 가장자리 부분과 핵심적인 부분의 명확한 정보 차이가 줄어들어 성능이 낮게 나온다고 보아도 될 것 같습니다. 논문에서는 이 한계를 해결하는 것을 future work로 제시합니다.


# [4] 마치며
형태로만 보면 정말 간단한 변형임에도 불구하고 논문의 흐름을 이해하기가 어려웠습니다. 덕분에 기하학 관련 공부를 많이 할 수 있었습니다. 아직 갈 길이 먼 것 같지만 관념적으로만 생각했던 파라미터 공간을 자세하게 살펴본 것이 저에게는 매우 좋은 경험이었습니다. 



# [5] 참고자료
- [[Paper] Layer Normalization](https://arxiv.org/abs/1607.06450)
- [[Post] An Overview of Normalization Methods in Deep Learning](http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)
- [[Post] An Intuitive Explanation of Why Batch Normalization Really Works](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)
- [[Post] Weight Normalization and Layer Normalization Explained](http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/)
- [[Post] Fisher Information Matrix](https://wiseodd.github.io/techblog/2018/03/11/fisher-information/)