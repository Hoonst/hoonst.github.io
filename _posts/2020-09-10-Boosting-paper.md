---
layout: post
title: "Boosting - Paper"
tags: [paper_review]
date: 2021-01-10
comments: true
typora-root-url: ../../hoonst.github.io
---
# A Short Introduction to Boosting

## 초록

Boosting은 어떤 학습 알고리즘이라도 성능을 향상시킬 수 있는 방법입니다. 즉, Boosting 자체가 학습 알고리즘이 아니라, 강화를 해주는 역할을 하는 것입니다.
본 논문에선,
  * Boosting 대표 알고리즘, AdaBoost 소개
  * Boosting의 기저 원리
    * Boosting은 Overfitting으로부터 안전하다.
    * Support Vector Machine과 Boosting과의 관계
를 소개할 것입니다.

## Introduction

본 논문에서 Boosting을 소개하기 앞서 상황을 통한 예시를 설명합니다. 이 예시를 뒤에서 수식을 설명함에 있어 매칭을 시키면서 진행하기에 안내하겠습니다.

### 상황 예시 - Horse Racing Gambler

한 **경마 도박꾼(이하 경마맨)** 이 자신의 도박 승률을 높이기 위하여, 경마 데이터(각 말들의 승률, 베팅률)을 통해 승리마를 예측하고자 합니다. 마치 프로젝트에 앞서 전문가에게 자문을 구하는 것처럼, **전문 도박꾼(이하 도박맨)** 에게 조언을 구하였습니다. **도박맨** 은 데이터가 없을 때는, 자신의 감을 통한 도박의 규칙을 설명하지 못하지만, 데이터가 있을 때는
* 최근 경주를 이긴 말들에 투자해라
* 배당률이 높은 말에게 투자해라

등 매우 당연하면서 광범위로 적용가능한 조언을 해주었습니다. 이를 논문 영어 표현에서는 "Rule of Thumb"이라고 하는데 사실 이 표현이 매우 알맞은 표현입니다.

![](/Boosting-Paper/rule_of_thumb.PNG#center)

> Rule of Thumb
* 엄격, 정확, 신뢰할 수 있기보단, 모든 상황에 대해 광범위하게 적용할 수 있는 원칙
* 이론보다 실제 경험을 바탕으로 쉽게 배우고 쉽게 적용 가능한 표준 절차

왜냐면 Boosting이 바로 이런 애매하고, 어리숙한 논리, 주장들을 모아모아 강한 주장으로 바꾸는 절차이기 때문입니다.
**도박맨** 의 조언들을 최대한 쓸모있는 조언으로 바꾸기 위해선 두 가지 문제에 봉착합니다.
1. **경마맨** 이 **도박맨** 에게 어떤 데이터를 보여줘야 좋은 조언을 해줄 것인가?
2. **도박맨** 의 조언을 하나의 쓸모있는 조언으로 바꾸는 방법은 무엇인가?

다시 정의하자면, **Boosting**이란,

> 여러 다소 부정확한 "Rule of Thumb"을 하나의 매우 정확한 규칙으로 바꾸는 과정

입니다.

## Backgroud
Boosting은 사실 "PAC" 모델에 기반해서 탄생했습니다.

## AdaBoost

1995년에 나타난 AdaBoost는 이전의 Boosting 알고리즘이 해결하지 못한 여러 문제들을 해결하여 두각을 나타냈습니다.

논문에서 제시하는 AdaBoost의 Pseudo Code는 다음과 같습니다. 복잡한 영어를 하나씩 풀어보겠습니다.
![](/Boosting-Paper/Pseudocode-for-the-boosting-algorithm-AdaBoost.png#center)

* Given: $(x_1,y_1),...,(x_m, y_m), x_i \in X, y_i \in Y = \{-1, 1\}$
 $x_i \in X, y_i \in Y = \{-1, 1\}$에서 X는 Input Data, Y는 Label입니다.
Label 같은 경우, 본 논문에서 주로 Binary를 다룬 뒤 Multi Class로 넘어갑니다.

  데이터셋은  $(x_1,y_1) ... (x_m, y_m)$와 같이 구성됩니다.

* Sample Weight $D_t$
AdaBoost는 Weak 또는 Base Algorithm을 반복해서 불러옵니다. 즉, Bagging에서는 병렬적인 처리를 진행하지만,
Boosting에서는 연쇄적으로 진행하는 것입니다. 여기서 Weight라는 단어로 인해 혼동이 일어나는데 Weight가 보통 파라미터 가중치를 의미하는데에 사용되기 때문입니다. 따라서 Training set 또는 Sample에 대한 Weight를 Sample Weight라 칭하겠습니다.

  Sample Weight: Weight of distribution on training example
  이는 Sample에 대한 가중치로서 하나의 Sample에 대하여 뽑일 수 있는 확률을 나타냅니다. Boosting 알고리즘을 사용할 때 Sample들을 Bootstrap으로 뽑아내기 때문에 복원 추출을 할 때 몇몇의 Sample들이 좀 더 높은 확률로 뽑일 수 있는 가능성을 열어두는 것입니다.

  한 time(t)의 Sample Weight 분포는 $D_t(i)$로 표현합니다.

  Initialization 단계에서는 이 가중치를 1/m(sample 갯수)로 정해놓으며, 차수가 지날 수록 오분류된 Sample들에 대한 가중치들이 증가하고, 정분류 된 Sample들의 가중치가 감소하는 방향으로 $D_t(i)$를 구성하게 됩니다. 이를 통해 문제가 있는 Sample에 대하여 Weak Learner들이 더 많이 검증할 수 있는 기회를 얻게 되는 것입니다.

* For t = 1...T: T번동안 Weak Learner들이 계산을 진행합니다.  

* Train weak Classifier using distribution $D_t$
보통의 학습 알고리즘의 목표는 X를 통해 Y를 예측하는 것입니다. 하지만 위에서 Boosting이 학습 알고리즘 자체가 아닌 '강화제'의 역할을 한다고 했기에, 본연의 목적은 예측이 아닙니다. Boosting의 Weak Learner들이 해야하는 임무는 ,

> $D_t$ 내에서 Weak hypothesis $h_t: X -> \{-1, +1\}$의 횟수를 찾는다

이는 달리 말하면, Sample Weight의 분포인 $D_t$에서 오분류한 Sample들의 Sample Weight를 한데 모아 더한 것이 바로 error가 되는 것입니다.

$\epsilon_t = Pr_{i \sim D_t}[h_t(x_i) \neq y_i]=$ $\sum_{i:h_t(x_i)\neq y_i}D_t(i)$

$\epsilon_t$는 매 시간 t의 Sample Weight 분포 $D_t$에서부터 만들어집니다.

실제로 Boosting을 사용할 때, 해당 $D_t$는 두 가지 방법으로 사용됩니다.
* Training Example에 $D_t$를 직접적 사용
* Training Example을 Sampling(Bootstrap)하기 위하여 간접적 사용 (직접적 사용이 안될 때)

##### 지금까지 살펴본 내용을 경마 비유로 대응

  * $x_i$ = 경마 데이터
  * $y_i$ = 결과 (승리마)
  * Weak hypotheses = rules of Thumb
  * $D_t$ = 룰을 만들때 사용한 데이터가 선택될 확률의 Sample Weight 분포  

* $\alpha_t$는 time t의 $h_t$의 중요도를 의미합니다.
$\alpha_t \geq 0$ $if$ $\epsilon_t \leq 1/2$이며,  $\alpha_t$는 $\epsilon_t$가 감소할수록 증가합니다. 이는 직관적으로 말이 되는 것이, Weak Learner, $h_t$의 에러가 감소한다면, 해당 알고리즘의 중요도는 커질 것이니 말입니다.

* $D_{t+1}$, How to Update
![](Boosting-Paper/update.png)
본 식의 목표로는 $h_t$에서 오분류된 것의 Sample Weight는 증가, 정분류된 것의 Sample Weight는 감소시키는 방향으로 진행합니다. 이로써 "정답을 내기 어려운" Sample들에 집중합니다.

* *final hypothesis* $H$
최종 H는 $T$개의 weak hypotheses들에 $\alpha_t$를 가중치로 얹어 다수결 투표를 진행하여 나타냅니다.
