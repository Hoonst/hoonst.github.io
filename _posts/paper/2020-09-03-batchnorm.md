---
layout: post
title: "Batch Normalization: Accelerating Deep Network Training by Reducing Covariate Shift"
description: "Sergey Ioffe, Christian Szegedy (ICML 2015)"
date: 2020-09-03
categories: paper
tags: [review, deeplearning, normalization]
image: 
---


# [1] 아이디어 제안 배경

## Covariate Shift

모델을 구성할 때에는 학습 데이터를 구성하여 학습을 합니다. 그 후 성능을 측정하기 위해 새로운 데이터를 사용합니다. 이 때 학습 데이터와 성능 측정용 평가 데이터의 분포가 다르다면, 모델의 성능은 하락하게 됩니다. 현실 세계의 많은 도메인 분야에서는 변수가 많기 때문에 이러한 현상이 자주 발생합니다. 이러한 현상을 **Dataset Shift**라고 부릅니다.  <br>

[참고한 글](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)에서 적절한 그림이 있어 함께 첨부합니다.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2017/07/07225817/low-score.png" width="70%">

**Covariate Shift**는 Dataset Shift의 한 종류입니다. Dataset Shift는 아래와 같은 종류로 구분됩니다.
1. 독립 변수(independent variables)에서의 변화: Covariate Shift
2. 목적 변수(target variable)에서의 변화: Prior Probability Shift
3. 독립 변수와 목적 변수 사이의 관계 변화: Concept Shift

즉 Covariate Shift는 학습 데이터와 평가 데이터의 "입력 변수 분포"가 다른 상황을 나타냅니다. 실제 세상에서 다른 종류의 Shift보다 빈번하게 나타나는 현상이라고 합니다. 도표로 보면 아래와 같습니다. 역시 같은 글에서 가져온 그림입니다.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2017/07/07230628/plot1.png" width="70%">



## Internal Covariate Shift
Internal Covariate Shift는 학습 단계의 내부에서 각 레이어(layer)의 입력으로 들어가는 값의 분포가 달라지는 현상을 말합니다. 위에서 설명한 Covariate Shift가 신경망 내부에서 각 층(sub-network)을 대상으로 발생하는 것으로 이해하시면 되겠습니다. <br>


논문에서 설명하는 흐름은 다음과 같습니다. 먼저 두 개의 층으로 이루어진 신경망을 가정합니다.


$$ l = F_2(F_1(u, \theta_1), \theta_2) $$


$F_1$과 $F_2$는 임의의 transformation이라고 가정하며, 두 번째 레이어의 입력값이 되는 $F_1(u, \theta_1)$을 $x$로 치환하면 아래와 같이 나타낼 수 있습니다.


$$ l = F_2(x, \theta_2) $$


이 형태는 입력값이 $x$이고 $F_2$ transformation을 적용한 단일 레이어와 같은 형태이기 때문에 Covariate Shift의 개념이 sub-network에도 적용될 수 있는 것입니다. <br> 

정리하자면 하나의 신경망이 두 개로 구성되어 있다고 가정할 때, 첫 번째 레이어에 입력되는 데이터의 분포와 두 번째 레이어에 입력되는 데이터의 분포가 다를 경우를 Internal Covariate Shift가 발생했다고 합니다.

*참고: 그러나 [이후 등장하는 논문](https://arxiv.org/abs/1805.11604)에서는 Internal Covariate Shift가 Batch Normalization이 잘 동작하는 이유와는 거리가 멀다고 주장하기도 합니다.*


## 깊은 신경망을 구성할 때 발생하는 문제
본 논문에서는 먼저 Stochastic Gradient Descent(SGD)를 할 때 두 가지를 특히 신중하게 결정해야 한다고 언급합니다. 첫 번째는 모델 파라미터의 초기 값, 두 번째는 최적화에 사용되는 하이퍼파라미터(특히 learning rate)를 신중하게 선택해야 한다는 것입니다. 학습 시 각 레이어의 입력값은 이전 레이어들의 파라미터에 영향을 받기 때문입니다. 그리고 신경망이 깊어질수록 영향을 받는(입력값이 변화하는) 정도가 더 커집니다. 그런데 Internal Covariate Shift가 발생하여 각 레이어 입력값의 분포가 변하게 된다면, 레이어마다 “새로운 분포에 적응”해야 한다는 문제점이 있습니다. 새 분포에 적응하는 데는 역시 비용이 발생하게 됩니다. <br>

신경망 학습의 전체적인 관점에서도 Internal Covariate Shift 현상은 문제를 일으킵니다. 바로 **Gradient Vanishing** 현상 때문인데요, 먼저 아래와 같은 레이어가 있다고 가정하겠습니다. 


$$ z = g(Wu+b) $$


여기서 $g$는 sigmoid 함수로, ${Wu+b}$를 $x$로 표현하고 수식으로 나타내면 ${g(x) = {1\over {1+exp(-x)}}}$로 나타낼 수 있습니다. Sigmoid 함수의 특성 상 입력값의 절댓값이 커지면 ${g\prime(x)}$는 0에 가까워집니다. 이 글에서 Gradient Vanishing 현상에 대해 더 자세히 설명하지는 않지만, 미분값이 0에 가까워진다는 것은 오차를 역전파할 때 전해지는 정보의 양이 거의 없어지는 것을 의미합니다. <br>

그런데 입력값 ${x=Wu+b}$는 가중치 $W$와 편향 $b$, 그리고 이전 레이어의 파라미터에 영향을 받습니다. 따라서 학습 도중 이 파라미터들이 지속적으로 변화한다면 $x$가 양 극단으로 치우쳐 Gradient Vanishing이 발생할 확률이 높아지게 됩니다. 이 또한 신경망이 깊어질수록 그 정도가 심해집니다. <br>

이 현상을 방지하기 위해 여러 방법이 제안되었습니다. 대표적으로는 아래와 같은 방법이 있었습니다.
1. Rectified Linear Units 함수: $ReLU = max(x,0)$
2. 신중한 가중치 초기화
3. 작은 learning rate

이렇게 좋은 방법들이 있지만, 본 논문에서는 입력값이 안정적인 분포를 갖는 것을 보장할 수 있다면 훨씬 빠른 학습을 할 수 있다고 말합니다. 그리고 제안한 방법론이 바로 **Batch Normalization** 입니다. 미리 언급하자면 다음과 같은 장점이 있습니다.

- 빠른 학습 가능
- 더 큰 learning rate 적용 가능
- 파라미터의 scale이나 초기값에 덜 의존함
- Regularization 효과가 있어 Dropout의 필요성 감소
- Gradient Vanishing 현상을 방지


# [2] 방법론

## Whitening
Internal Covariate Shift 현상을 없애기 위해 생각할 수 있는 가장 간단한 방법은 **whitening**입니다. Whitening은 평균을 0, 분산을 1로 변환시키고 decorrelation을 적용하는 기법을 말합니다. 각 레이어의 입력값에 whitening을 적용한다면 일정한 분포를 갖게 될 것입니다. <br>

그렇다면 자연스럽게 학습의 각 step 또는 일정 주기마다 whitening을 적용하는 모습을 상상할 수 있습니다. 그러나 위 방법은 몇 가지 문제점을 갖고 있습니다. 지금부터 설명하는 내용은 간단하게만 정리합니다. <br>

### 문제점 1: Gradient Descent 방법론과의 충돌
Whitening은 최적화 단계에서 번갈아서 적용될텐데, gradient descent 방법은 **normalization term을 업데이트하는 방향**으로 파라미터를 업데이트하게 됩니다. 먼저 입력값 $u$에 $b$라는 편향을 더한 $x = u+b$를 가정합니다. 그리고 normalization을 위해 다음과 같이 변형할 수 있습니다.


$$ \widehat{x} = x-E[x] $$


여기서 $b$를 업데이트할 때 $x$를 풀어서 다음과 같은 식으로 나타낼 수 있습니다.


$$ u + (b+\vartriangle{b}) - E[u + (b+\vartriangle{b})] = u + b - E[u+b]$$


정리하자면 $b$를 업데이트하고 normalization을 적용한다면 loss가 바뀌지 않습니다(이 현상에 대해서는 경험적으로 찾았다고 언급합니다).

### 문제점 2: 연산 비용 부담
역전파를 위해 Jacobian Matrix를 계산하려면 whitening이 적용된 부분의 공분산행렬과 그것의 inverse square root를 구해야만 합니다. 이는 많은 연산량을 요구하여 간단하게 적용하기에는 무리가 있습니다. 따라서 쉽게 미분가능하면서도 (normalize를 위한)전체 데이터셋을 신경쓰지 않아도 되는 방법이 필요했습니다.

## Batch Normalizing Transform
위에서 언급된 문제를 해결하기 위해 본 논문에서는 두 가지 방법을 제안합니다. 


### 첫 번째 제안: normalizing each feature
첫 번째는 각 feature를 **독립적**으로 간주하고, 각각을 normalize하는 방법입니다. 즉 $d$차원 입력값 $X=(x^{(1)} \cdots  x^{(d)})$의 각 요소를 아래와 같이 normalize하는 것입니다.


$$ \widehat{x}^{(k)} = { {x^{(k)} - E[x^{(k)}] } \over { \sqrt{Var[x^{(k)}]} } } $$


하지만 단순하게 각 입력값을 normalize하는 것은 올바른 방법이 아닙니다. Sigmoid 함수의 입력값을 normalize하게 된다면 표현 범위가 작아져 비선형성이 제대로 적용되지 않을 수도 있기 때문입니다. 이를 보완하기 위해 논문에서는 linear transformation과 아래의 규칙을 도입합니다. <br>

<center><i>the transformation inserted in the network can represent the identity transform </i></center> <br>

즉, linear transformation을 적용한 후에도 원래의 representation을 다시 복원할 수 있어야 한다는 말입니다. 이 규칙을 만족시키기 위해 새로운 파라미터인 $\gamma^{(k)}$와 $\beta^{(k)}$로 scaling과 shifting을 적용합니다. 새로운 파라미터들은 학습을 진행하면서 함께 업데이트됩니다. 표현하자면 아래의 식과 같습니다.


$$ y^{(k)} = \gamma^{(k)} \widehat{x}^{(k)} + \beta^{(k)} $$


위 식에서 $\gamma^{(k)} = \sqrt{Var[x^{(k)}]}$ 그리고 $\beta^{(k)} = E[x^{(k)}]$이 된다면 원래의 형태 역시 보존할 수 있습니다.


### 두 번째 제안: mini-batch setting
Normalization을 위해서는 나누어 줄 전체 데이터셋이 필요합니다. 그런데 일반적으로 사용되는 Stochastic Optimization에서 전체 학습 데이터셋을 사용하는 것은 비효율적입니다. 따라서 본 논문에서는 mini-batch 단위로 normalization을 적용하는 방법을 제안합니다.


위 방법들을 포함한 Batch Normalization의 알고리즘은 아래와 같습니다.

<img src="/assets/figures/batchnorm.PNG" width="70%">

Normalization이 적용된 각 입력값 $\widehat{x}$는 sub-network를 거쳐 신경망 전체에 고정된 평균과 분산을 갖도록 합니다. 이 방법이 적용된 상태의 오차 역전파는 아래와 같은 과정으로 진행됩니다. 이는 곧 Batch Normalization이 미분가능한 형태이면서도 신경망에 normalization 효과를 주는 방법임을 보여줍니다. 

<img src="/assets/figures/bn_backprop.PNG" width="70%">




## Train/Inference with BN
Batch Normalization을 신경망에 적용하려면 간단히 레이어의 입력값을 $x$ 대신 $BN(x)$으로 변환하면 됩니다. 그런데 학습 시와 추론 시 적용하는 Normalization의 형태가 조금 다릅니다. 추론 시에는 mini-batch로 normalize할 필요가 없습니다. 또한 추론 시 mini-batch로 normalize를 하게 된다면 출력값이 입력값에만 의존하는 것이 아니기 때문에 deterministic한 결과를 얻기가 어렵습니다. 그래서 미리 샘플링하여 계산한 평균과 분산을 사용하여 normalization을 적용합니다. 식으로 표현하면 아래와 같습니다. 참고로 $\epsilon$는 분모가 0이 되는 것을 방지하는 아주 작은 수입니다.


$$ \widehat{x} = {x-E[x] \over \sqrt{Var[x] + \epsilon}} $$


Batch Normalization을 적용하여 전체 신경망을 학습하는 알고리즘은 아래와 같습니다.


<img src="/assets/figures/bn_train.PNG" width="70%">

## BN의 장점
Batch Normalization은 각 레이어의 nonlinear 활성화 함수 이전에 적용합니다. 아래와 같은 transformation이 있다고 가정하겠습니다.


$$ z = g(Wu+b) $$ 


여기서 $g$는 Sigmoid나 ReLU 등의 활성화 함수이고, BN은 $u$가 아닌 $Wu+b$에 적용됩니다. 그 이유는 $u$ 역시 이전의 non-linearity가 적용된 출력값이기 때문입니다. 이제 Batch Normalization 방법론의 장점에 대해 알아보겠습니다.

### 장점 1: 큰 learning rate 사용 가능
이전의 깊은 신경망에서 큰 learning rate를 사용하면 Gradient Vanishing 또는 Exploding 현상이 발생하거나 변변찮은 local minima에 수렴하는 경향이 있었습니다. Batch Normalization은 Internal Covariate Shift 현상을 방지하기 때문에 학습 시 saturate하는 현상이 발생하지 않습니다. <br>

또한 파라미터의 스케일에 영향을 받지 않습니다. Learning rate가 크면 파라미터의 스케일이 증가합니다. 이는 역전파 시 gradient를 증가시켜 Gradient Exploding이 발생할 수 있습니다. 하지만 BN을 적용할 경우 오히려 가중치가 커 지면 gradient가 작아집니다. 따라서 파라미터의 업데이트를 안정화시킬 수 있습니다.

### 장점 2: 모델 regularization 효과
Batch Normalization을 사용하여 학습을 진행할 경우, mini-batch에 따라 출력값이 deterministic하지 않게 산출됩니다. 이는 신경망을 일반화하는 효과와 같으며, 따라서 기존에 사용되던 Dropout 기법을 사용하지 않거나 줄여도 무방하다고 언급합니다.

# [3] 실험
실험에 대한 설명은 아래의 그림을 첨부하는 것으로 대신하겠습니다.

<img src="/assets/figures/bn_exp.PNG" width="70%">

## BN Network 잘 사용하기
Batch Normalization은 ImageNet Classification 문제에서 뛰어난 성능을 보였습니다. 그런데 본 논문에서는 단순히 BN을 추가하는 것만으로는 그 장점을 온전히 살릴 수 없다고 말합니다. 그리고 다음과 같은 방법을 권장합니다.
1. Learning rate 증가시키기
2. Dropout 제거하기
3. $L_2$ 가중치 정규화 제거하기
4. Learning rate decay 가속하기
5. (Inception 등의 모델에서 사용된)Local Response Normalization 제거하기
6. 학습 데이터를 더 섞기(같은 mini-batch에서 항상 다른 example이 등장하도록)
7. Photometric Distortion을 줄이기
   - BN 신경망이 빠르게 학습하면서 각 학습 example을 더 적게 보기 때문에 최대한 왜곡이 적은 실제 이미지와 가까운 example을 선호합니다.

# [4] 마치며
2020년 현재 [PyTorch에서 코드 한 줄이면 손쉽게 사용](https://pytorch.org/docs/stable/nn.html#normalization-layers)할 수 있는 Batch Normalization에 대해 알아보았습니다. 반복적인 실험을 통해 좋은 성능을 낸 논문도 물론 좋지만, 핵심적인 아이디어를 제안한 논문을 읽으면 저자의 논리 흐름을 읽는 재미가 있는 것 같습니다.


논문에서는 BN을 비선형 활성화함수 이전에 적용해야 한다고 말하지만, 예전에 활성화함수 이후에 BN을 사용하는 코드를 본 적이 있었습니다. 경험적으로 찾아낸 것이 아닌가 싶었는데, [BN을 적용하는 시점에 대한 토론](https://stackoverflow.com/questions/47143521/where-to-apply-batch-normalization-on-standard-cnns)이 이루어진 적이 있었고 [활성화함수 이후에 적용했을 때 더 높은 성능을 보이는 경우](https://github.com/gcr/torch-residual-networks/issues/5)도 있었습니다. 이를 보고 Internal Covariate Shift가 BN의 성능 향상과 큰 연관이 없다는 것을 설명하는 하나의 사례가 되지 않을까 생각하였습니다.


본 논문의 Future Work에서는 해당 방법론을 RNN 기반의 모델에 적용하는 방향을 제시합니다. 자연어처리에서 많이 쓰이는 Transformer 모델에 Layer Normalization 기법이 사용되는데, 그 논문에서는 어떠한 논리를 전개할 지 기대가 됩니다.


# [5] 참고자료
- [[Paper] Batch Normalization: Accelerating Deep Network Training by Reducing Covariate Shift](https://arxiv.org/abs/1502.03167)
- [[Post] Covariate Shift – Unearthing hidden problems in Real World Data Science](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)



