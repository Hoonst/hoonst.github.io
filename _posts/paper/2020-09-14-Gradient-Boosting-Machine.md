---
layout: article
title: "Gradient Boosting Method"
tags: paper_review
mathjax: true
article_header:
  type: cover
  image:
    src: /assets/BaggingPredictors/2020-09-03-Bagging-Predictor-57f42fef.png
---

# Gradient Boosting Machine 정리
Boosting을 알면 GBM은 자동적으로 이해가 가능한 알고리즘인 줄 알았습니다. 왜냐면 Gradient는 알고 있는 개념이며 이것이 이전에 다룬 Boosting과 합체하여 사용되는 느낌이었기 때문입니다. 하지만 논문의 수식으로는 해당 '합체 과정'을 명확하게 파악할 수 없어 본 영상을 시청 후 기록하고자 합니다.

[![](http://img.youtube.com/vi/3CC4N4z3GJc/0.jpg)](http://www.youtube.com/watch?v=3CC4N4z3GJc)

선이수 지식: Decision Tree / AdaBoost / Bias & Variance

### 예시 데이터

* 키와 선호 색깔과 성별로 (무게)를 맞춰라!
* 가장 Common한 방식으로 Gradient Boost를 설명
* 종속변수가 수치형이라면 Gradient Boost Regression
* 해당 Regression은 일반 Linear Regression과 다름

![](assets/2020-09-14-Gradient-Boosting-Machine-e363a619.png){:.border.rounded :height="300px" width="300px"} ![](assets/2020-09-14-Gradient-Boosting-Machine-526078a5.png) {:.border.rounded :height="300px" width="300px"}

이전에는 AdaBoost가 GBM의 일원인 줄 알았습니다. 하지만 이 둘은 공통점이 있지만 다른 것이므로, 이 사실을 명확히 알고 학습해야 합니다.

## AdaBoost vs GBM

### GBM

먼저 initial output, 사례에서 Weight를 만들어 냅니다. 이는 모든 Sample에 대한 initial guess라고 할 수 있습니다. 연속형 변수를 예측하는 경우이니, 최초 값은 평균이 될것입니다.

그 후, GBM은 이전 Tree에서 겪은 에러를 기반으로 Tree를 만듭니다. Stump으로만 트리를 구성했던 AdaBoost와는 다르게 GBM은 Tree 그 자체를 만듭니다. 물론 트리 사이즈를 조절하지만 말이지요. 예시로는 4개의 잎까지만 만드는 것으로 하겠습니다. 실제로는 8~32정도의 수로 정한다고 합니다.

즉, AdaBoost와 같이, GBM은 이전의 에러를 기반으로 고정된 트리를 만들지만, Stump보다 더 큰 트리를 만듭니다.

또한 GBM은 Adaboost와 같이 트리를 스케일하지만, 모든 트리를 같은 정도로 조절합니다.

그럼 본격적으로 예시를 통해 절차를 하나씩 짚어보도록 하겠습니다.

1. Initial Output
현재 Weight 열의 평균은 71.2입니다. 이를 사용하여 모든 행의 결과를 71.2로 예측합니다. 하지만 여기서 멈추는 것은 말이 안되겠죠!
도출된 에러를 통해 다음 트리를 구성합니다.

이때 에러는 **관측값 - 예측값** 으로 계산합니다.
그리고 이 에러를 'Pseudo Residual'이라고 칭하고 저장합니다. 본디 Residual은 선형 회귀를 시행할 때 사용하는 말인데, GBM Regression을 하고 있음을 상기시키기 위하여 다른 단어를 사용하고자 합니다. 즉 다음 사진과 같이 구성합니다.

![](assets/2020-09-14-Gradient-Boosting-Machine-10fb83dd.png){:.border.rounded :height="300px" width="300px"}

2. Residual을 예측하라!

![](assets/2020-09-14-Gradient-Boosting-Machine-ee0a3ead.png){:.border.rounded :height="300px" width="300px"}

이제 갖고 있는 feature들을 통해 Pseudo Residual을 예측합니다. 아니... Residual을 예측한다고? 뭔가 이상하다 싶지만 계속 진행하면서 이 껄끄러움을 해소해보겠습니다.

![](assets/2020-09-14-Gradient-Boosting-Machine-6366aa93.png){:.border.rounded :height="300px" width="300px"} ![](assets/2020-09-14-Gradient-Boosting-Machine-cb064dcf.png){:.border.rounded :height="300px" width="300px"}

Residual을 예측하면 다음과 같이 트리를 구성할 수 있습니다. 우리는 4개의 잎까지만 만들기로 설정했지만, 행은 8개입니다. 즉, 행보다 잎이 적은 꼴이며, 잎을 공유하고 있는 값들도 존재합니다. 이럴때는 평균치를 구하여 대체합니다.

이후, 최초의 Initial 값으로 설정한 71.2에 해당 Residual의 예측값들을 더합니다.

![](assets/2020-09-14-Gradient-Boosting-Machine-3346519d.png){:.border.rounded :height="300px" width="300px"}

[Gender=F] > [Color not Blue] > 16.8의 잎을 Initual 값 71.2에 더하게 되면 88이 나옵니다. 하지만 해당 88은 원래의 값과 동일하게 됩니다. 원래라면 오차가 0이니 좋아라 하겠지만 GBM에서는 반기지 못하는 이야기 입니다. Overfitting으로 인해 낮은 편향은 이룰 수 있겠지만, 높은 분산이 나타나게 될 것이기 때문입니다.

GBM은 이를 방지하기 위하여 Learning Rate(0~1)로 트리의 힘을 조절합니다. 예시에서는 0.1로 설정하여 계산해보겠습니다

> 71.2 + (0.1 x 16.8) = 72.9

이로써 트리의 예측값은 72.9, 즉 88과 많이 다르게됩니다. 하지만 생각해보면 71.2보다는 정답에 가까워졌습니다. GBM의 저자 Friedman은

"올바른 방향으로의 조그마한 걸음들이 모여 검증 데이터에서 더 좋은 결과를 가져온다" 라고 언급했습니다. 즉 낮은 분산을 이룩하는 것이지요.

3. 그럼 이제 새로운 예측값을 통해 다시 Residual을 예측해봅시다

**Residual = (88 - (71.2 + 0.1 x 16.8)) = 15.1**

![](assets/2020-09-14-Gradient-Boosting-Machine-830758a5.png){:.border.rounded :height="300px" width="300px"}

이로써 첫번째 트리를 구성할 때의 Residual과 두번째 Tree의 Residual이 달라졌음을 알게 되었습니다.

NOTE: 지금 예시에서는 트리의 가지가 매번 같습니다만, 실제로는 매번 달라질 수 있습니다.
