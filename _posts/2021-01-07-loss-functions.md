---
layout: post
title: "Loss Functions"
description: "Loss Function 정리"
tags: [assignment]
date: 2021-01-07
comments: true
typora-root-url: ../../hoonst.github.io
---

# 예측모델 과제 #2

본 포스트는 고려대학교 산업경영학과 김성범 교수님의 '예측모델' 수업 과제의 일환으로 작성된 포스트입니다.

목차

* Loss Function의 중요성
* Max Error
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
  - (R)MSE와 MAE 언제 사용해야 하나?
  - MAE LOSS / MSE LOSS 미분
  - MAE = Median / MSE = Mean을 탐색하는 과정?
* MAPE (Mean Absolute Percentage Error)
* MPE (Mean Percentage Error)
* Huber Loss
* Quantile Loss
* $R^2$ & Adjusted $R^2$

### Loss Function의 중요성

---

Loss Function의 대표적인 정의는 "실제값과 모델의 예측값의 차이(Error)를 비교하여 모델의 성능을 평가하는 지표" 로 나타낼 수 있다.

​	머신러닝에서 모델을 구축할 때는 "Gradient Descent"라는 대표적을 과정을 따르며, 이 때, 미분을 하여(Gradient) 아래로 내려가는 과정(Descent)의 대상은 Loss Function이다. Loss Function의 최소값을 향하여 내려가는 것을 목표로 하다보면, 모델의 에러가 최소화되는 지점을 찾을 수 있다. 즉 비유적으로 표현하자면, Loss Function은 자녀를 가르치는 부모의 지도라고 할 수 있으며, 자녀가 옳지 않을 길로 가는 것을 나타내어, 옳지 않은 행동을 최소화하는 과정을 거치게 된다. 

​	Loss Function은 단일한 것이 아니라 매우 다양하며, 때에 따라 적용해야 하는 Loss Function이 다르다. 마치 세상에는 다양한 자녀와 부모가 있어 훈육의 방식이 다른 것 처럼 말이다. 통계 및 머신러닝에서는 '다양한 상황'의 예시는 다음과 같다. 

* Outlier를 어떻게 대우할 것인가?
* 어떤 머신러닝을 사용할 것인가?
* Gradient Descent가 가능하도록 미분이 가능한가?

등의 다양한 사례가 있다. 때에 맞는 Loss Function을 정하고 사용해야 모델을 올바른 길로 인도하기에 적합하다.

머신러닝의 대표적인 과제는 Classification과 Regression이 존재한다. 하지만 본 과제에는 Regression에 대한 성능을 측정할 수 있는 Measure, 즉 Loss Function을 정리해보고자 한다. 



### MAX ERROR

---

$Max Error(y, \hat y) = max(|y_i, \hat y_i|)$

**<정의>**

Max Error는 에러의 최악의 경우를 나타낸다. 만약 완벽하게 실제 현상에 걸맞는 모델을 만들었다면 Max Error는 0이 되겠지만, 현실적으로 일어나기 어려운 일이다. 즉, 극단적인 모델의 에러를 나타내기 위해서 존재는 하지만 실용성은 적다. 



### MAE(Mean Absolute Error)

---

$MAE(y,\hat{y}) = \frac{1}{m}\sum_{i=1}^{n}|y_i - \hat{y_i}|$

**<정의>**

MAE는 가장 **기본적**이고, 가장 **직관적**인 평가 지표이다. 왜 그럴까?

모델의 성능을 **'모델이 해야할 일을 잘 하는지'**에 대한 평가라면, 정답을 가장 잘 맞추는 모델의 성능이 제일 높은 것이다. 이런 성능을 측정하기 위해선 실제 정답과 예측한 값의 차이를 알면 되고 이는 단순히 정답과 예측값 차의 절대값을 씌운 값으로 표현할 수 있다. 그리고 이를 모든 데이터 포인트에 다하여 계산한 후 평균을 내게 되면 MAE가 된다. 

아래의 그림같이, **파란색 점**이 정답, **초록색 선**이 예측값일 때, 이들의 절대 차이들(빨간 선)을 모두 더함으로써 MAE를 구하는 것이다. 

**<특징>** 

MAE는 모든 데이터에 대하여 공평하게 대우한다. 즉, Outlier같이 MAE에 지나치게 큰 영향을 미치는 데이터도 공평하게 반영한다. 아래의 그래프와 같이 절대 차이, 즉 Loss를 빨간 선으로 나타냈을 때, 본연의 값 그대로 MAE 식에 반영이 되어 계산이 된다. 즉, 값만이 포함되며, 어떤 추가적인 계산이 이루어지지 않는다는 뜻에서 공평하게 대우한다는 뜻이다. 

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200922125752944.png" alt="image-20200922125752944" style="zoom:33%;" />

### MSE(Mean Squared Error)

---

$MSE(y, \hat y) = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2$

**<정의>**

MSE는 MAE와 매우 유사하지만 평균을 내기 전에 정답과 예측값의 차이에다가 제곱을 해준다. 이로써 똑같은 모델을 평가함에 있어도 MSE가 MAE보다 항상 클 수 밖에 없다. 아래 그림처럼 Error의 값이 절대값 차이에서 사각형의 꼴(제곱)로 증가하는 모습에서, Error가 클 수록 해당 Error를 더 영향력 있게 반영하겠다는 뜻이 된다. 이는 만일 Outlier가 존재하게 되면, 없을 때와 비교했을 때, 비약적인 MSE 차이를 보일 수 있다. 

따라서 Outlier를 어떻게 취급할 것인가에 따라서 MAE냐 MSE냐를 선택해야 한다. 만일 Outlier가 성능 평가 지표에 영향을 주는 것을 원하면 MSE, 그렇지 않으면 MAE를 사용해야 한다. 

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200922125819912.png" alt="image-20200922125819912" style="zoom:33%;" />

### RMSE(Root Mean Squared Error)

---

$RMSE(y, \hat y)= \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}$

실제값 $y_j$ 와 예측값 $\hat{y}_j$ 의 차이 제곱의 평균(MSE)에 제곱근을 씌워 MSE를 계량화 한 것이다 (Residual).

이는 MSE의 확장판이다. MSE는 제곱을 취해줬기 때문에, 원래의 단위에서 많이 벗어나게 된다.

예를 들어, $y_j$가 100, $\hat{y}_j$ 가 0이고, 데이터가 하나라면, MSE는 $100^2 =10000$ 이 된다. 즉, 차이의 단위가 백 단위에서 만 단위가 되어버린 것이다. 이는 모델간의 비교에는 영향이 없을 수 있으나, 해석의 난해함을 불러 일으킬 수 있다. 이런 모습은 마치 분산에 제곱근을 씌워 표준편차를 만들고, 그리고 공분산에 제곱근을 씌워 상관계수를 만들어 단위를 맞추는 양태와 같다. 

**(R)MSE와 MAE, 언제 사용해야 하나?**

MAE와 (R)MSE는 매우 비슷하면서도 다르며, 각자의 Loss Function으로서의 특징이 명확하다. 따라서 각자에게 어울리는 경우를 한번 정리해보고자 한다. 

* MAE
  * Outlier가 Loss Function에 주는 영향을 최소화하기 위해
  * Model들의 상호 비교를 하기 위해
* (R)MSE
  * Outlier가 Loss Function에 주는 영향을 반영하기 위해
  * 절대값을 씌운 MAE와 다르게, 제곱의 꼴을 갖고 있는 (R)MSE는 Continously하게 미분이 가능하다(0에서 미분 가능). 이런 이유로 미분을 사용하여 파라미터를 업데이트 해나가는 머신러닝 알고리즘들이 MAE보다 MSE를 많이 이용하는 이유이다. 
    아래의 이미지는 평가 Metric 함수를 표현한 것이다. Squared, 즉 (R)MSE 계열은 $x=0$일때 미분이 가능하지만, Absolute인 MAE는 미분이 불가능하여 계산을 어렵게 만든다. 

<img src="https://i.stack.imgur.com/vXMgz.png" alt="Some different loss functions" style="zoom: 25%;" />

**MAE LOSS / MSE LOSS 미분**

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200925131841447-0009942.png" alt="image-20200925131841447" style="zoom:33%;" />

MAE의 그래프는 미분의 값이 같은 방향 내에서 항상 같다. 즉, Loss 값이 작아지더라도 그에 대한 미분값이 계속 유지가 된다는 뜻이다. 이는 좌측 그림에서 나타나듯이 최적의 값을 지나치는 경우를 발생시킬 수 있다. 이를 방지하기 위하여 Learning rate를 Dynamic하게 설정할 수도 있지만, MSE를 사용하는 것이 낫다. MSE의 미분값은 값이 Loss가 점점 작아질수록 미분값도 작아지고 그에 따라 최소값에 다다르기 용이하기 때문이다. 

**MAE = Median / MSE = Mean을 탐색하는 과정?**

![image-20200925012503317](/assets/2021-01-07-.assets/image-20200925012503317.png)

MAE 결과의 특징으로, 예측값 $\hat y$가 $y$의 중앙값을 향해 훈련이 된다고 하며, MSE 결과의 특징으론  예측값 $\hat y$가 $y$의 평균을 향해 훈련이 된다고 한다. 직관적으로 생각해보면, Outlier에 대한 민감도가 이런 결과를 낳는다고 생각한다. 중앙값은 Outlier에 대하여 관용적이고, 평균은 영향을 많이 받기 때문에 같은 특성은 갖고 있다고 생각한다. 하지만 검색을 통해 찾아보아도 명확한 설명을 제공하는 곳이 없었기에 조력을 받은 뒤에 정리하고자 한다. 



### MAPE (Mean Absolute Percentage Error)

---

$MAPE = \frac{100\%}{n}\sum_{t=1}^{n}\left |\frac{y_t-\hat y}{y_t}\right|$

MAPE는 MAE를 확률로서 표현한 것이다. 확률적으로 접근한 것만 빼면, MAE가 모델의 **평균적인 에러**를 나타내는 것과 같이, MAPE도 같은 의미를 갖고 있다. 

##### MAE와 MAPE의 공통점

* 모델의 **평균 에러**: 예측값이 실제값과 평균적으로 얼마나 떨어져 있는가?
* 절대값이기에 이상치에 대하여 Robust하다.

##### MAPE의 ''치명적'' 단점: 나누기의 함정

MAPE는 계산 절차에 나눗셈이 포함되며, 이는 여러 문제를 낳는다.

* 실제값 $y$ 가 0일 경우, division by zero 문제가 나타난다.
* 실제값 $y$ 가 0은 아니지만 매우 작게 되면, 분모가 매우 작으니 지나치게 MAPE가 커질 수 있다.
* MAPE는 실제값보다 작을 경우를 더 선호하게 될 수 있다.
  <img src="/assets/2021-01-07-Loss-Functions.assets/image-20210107180106198.png" alt="image-20210107180106198" style="zoom:67%;" />

​	초록색의 경우: 
​	$\frac{100\%}{n}\sum_{t=1}^{n}\left |\frac{20-10}{20}\right|$ =100% * 1/2 = 50%

​	빨간색의 경우:
​	$\frac{100\%}{n}\sum_{t=1}^{n}\left |\frac{10-20}{10}\right|$ =100% * 2/1 = 200%



##### MAPE의 해석

MAPE가 15라면, 실제값과 예측값이 평균적으로 15% 차이가 난다는 의미이다.



### MPE (Mean Percentage Error)

---

$MPE = \frac{100\%}{n}\sum_{t=1}^{n}\left (\frac{y_t-\hat y}{y_t}\right)$

MAPE와 꼴은 완전 똑같으나 절대값이 사라진 경우이다. 

절대값이 사라짐으로써 양수값과 음수값을 한 값에 모아 표현할 수 있는데 이는 장점과 단점을 모두 갖고 있다.

**장점**

MPE의 값은 부호로 나타낼 수 있으므로, 만약 양수이면 적합시킨 모델이 실제값보다 과대추정을 한다고 할 수 있으며, 음수이면 과소추정을 한다고 판단할 수 있어, 에러의 추세를 알 수 있다. 이는 여러 모델을 **앙상블**할 때 활용될 수도 있는데, 그 이유는 과소추정 / 과대추정만 하는 모델들끼리 모아서 앙상블을 하는 대신, 둘을 적당히 균일하게 분배하여 사용하면 상호 보완을 통해 예측력을 높일 수 있기 때문이다. 

**단점**

양수와 음수의 에러가 서로 빼고 더하는 과정이 포함되어, 모델의 예측력에 대한 평가를 할 수가 없다. 예를 들어, 데이터 1~10번까지의 에러의 총합이 -100, 11~20번까지의 에러의 총합이 100이라면 상충되어 0이 될 것이니, 에러가 많아도 판단을 할 수 없다. 

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200922125920265.png" alt="image-20200922125920265" style="zoom: 33%;" />

### Huber Loss

---

$L_{\delta}(y, f(x))
\begin{cases} \frac{1}{2}(y-f(x))^2 & \mbox for |y-f(x)| \leq \delta\\ \delta|y-f(x)| - \frac{1}{2} \delta^2 &\mbox otherwise.\end{cases}$

Huber Loss는 Loss ($|y-f(x)|$)가 $\delta$보다 작을 경우에는 MSE로, 반대의 경우에는 MAE로 계산한다. 이로써 MSE보다는 Outlier에 대하여 덜 민감하며, MAE가 불가능했었던, '0에서의 미분'이 가능하게 된다. 즉 기본적으로는 MAE의 꼴을 갖지만 에러가 작아질 때는 2차 꼴, MSE가 된다. 

Huber Loss에서 이상치에 대해서 덜 민감할 수 있는 이유는, MSE로서 존재할 수 있는 영역을 정하기 때문이다. 그런데 이상치란 보통 매우 크거나 매우 작은 값을 의미하므로, 0 주위에 포진해있을 가능성이 적다. 따라서 해당 부분만 2차식의 이점을 취하고 나머지는 1차식으로 계산하면 문제를 피해갈 수 있다. 

Huber Loss에서는 $\delta$로써 MSE의 범위를 정하는데, 이는 이상치로 정의할 범위를 정하는 것과 같다. $\delta$보다 큰 Outlier는 MAE으로 Loss가 정해질텐데, MAE는 이상치에 영향을 받지 않을 것이다. 

하지만 이 방법의 단점은 hyper parameter인 $\delta$를 반복적으로 직접 찾아 나서야 한다는 것이다. 

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200925023136240-0015259.png" alt="image-20200925023136240" style="zoom:50%;" />

### Quantile Loss

---

$L_\gamma(y,y^p) = \sum\limits_{i=y_i<y_i^p}(\gamma -1) |y_i-y_{i}^p|  +  \sum\limits_{i=y_i \geq y_i^p}(\gamma) |y_i-y_{i}^p|$

분위수 손실이라 불리는 Quantile Loss는 점추정보다는 구간추정에 어울리는 Loss이다. 

MSE와 같이 제곱을 사용하는 Loss는 잔차들이 균일한 분산을 가질 것이라는 가정으로 진행한다. 따라서 이런 가정을 위배하는 회귀식의 결과를 믿기는 어렵다. 예를 들어 살펴보자.

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200925031317173.png" alt="image-20200925031317173" style="zoom:67%;" />

좌측의 그래프는 잔차의 분산이 일정하다. 하지만 우측의 잔차의 분산은 $x$가 증가할수록 증가한다. 즉, 분산이 일정치 않다는 뜻이다. 이런 경우에는 단순 회귀식으로 예측했을 때, 처음에는 데이터의 패턴과 일치하지만 $x$가 커질수록 오차가 매우 커지는 것을 볼 수 있다. 그렇다고 비선형 함수나 트리 기반 모델이 해결해줄 것이라고 확신할 수 없다. 

위의 데이터에 Quantile Loss를 통해 모델을 적합시키면 다음과 같은 결과를 얻을 수 있다. 

<img src="/assets/2021-01-07-Loss-Functions.assets/image-20200925031657902.png" alt="image-20200925031657902" style="zoom:67%;" />

점선이 두 개 존재하는데 이는 $\gamma$를 0.05 설정하고, 각기 0.05 분위수, 0.95 분위수 회귀식을 나타낸 것이다. 즉, 우리의 예측이 저 점선 사이에 존재할 것이라는 모델을 나타낸 것이다. 

### $R^2$ & Adjusted $R^2$

---

$R^2$ 과 Adjusted $R^2$ 는 '설명'의 수단으로서 쓰인다. 이는 사용한 또는 선택한 독립변수들이 종속변수의 분산을 얼마나 잘 설명하는지 나타내준다. 

먼저 $R^2$부터 설명해보겠다.

$\hat R^2 = 1 - \frac{\sum_{i=1}^{n}(Y_i-\hat Y_i)^2}{\sum_{i=1}^{n}(Y_i-\bar Y_i)^2}$

분자는 MSE, 분모는 Y의 분산을 나타내고 있다. 모델의 MSE가 높으면 $R^2$가 낮아지고 이는 모델의 성능이 나빠짐을 시사한다. 

$R_{adj} = 1-[\frac{(1-R^2)(n-1)}{n-k-1}]$

Adjusted $R^2$는 기존의 $R^2$와 같이 설명의 수단으로 활용되지만, 모델의 수식을 약간 변형시킨다.  그리고 항상 $R^2$보다 작거나 같다.

* **n**: 관측치 갯수
* **k**: 예측 변수 갯수

**Adjusted $R^2$를 왜 사용해야 하나?**

당연히 기존의 $R^2$에 존재하는 문제를 해결하기 위해 사용한다. 기존의 $R^2$는 모델의 변수가 증가하게 되면 설명력이 증가한다고 판단한다. 그리고 이 증가는 해당 변수가 결과를 예측하는데 좋은 변수인지 아닌지에 상관없이, 무조건 상승하게 된다. 

**$R^2$ 계열과 (R)MSE의 비교**

(R)MSE는 모델의 에러를 나타내는 '양'이다. 즉, '양' 자체로는 모델의 성능을 평가하기는 어려우며, 항상 다른 모델 또는 Baseline과의 비교를 통해 성능의 우위를 평가한다. 

$R^2$는 이와 다르게 모델 자체의 성능을 나타낼 수 있다. 선택한 변수들로 나타낸 예측값과 실제값을 비교함으로써 현재의 모델의 탄탄함을 평가할 수 있는 것이다. 이는 모델 평가와 동시에 변수 선택에서도 사용될 수 있다.

### 참고자료

---

* Choosing the Right Metric for Evaluating Machine Learning Models - Part1
  https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4

* Tutorial: Understanding Regression Error Metrics in Python
  https://www.dataquest.io/blog/understanding-regression-error-metrics/

* Deep Quantile Regression
  https://towardsdatascience.com/deep-quantile-regression-c85481548b5a

* L1 vs L2 Loss Function
  http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/

* 5 Regression Loss Functions All Machine Learners Should Know
  https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

* Huber Error | Loss Functions
  https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/

* Quantile loss function for machine learning
  https://medium.com/@gobiviswaml/huber-error-loss-functions-3f2ac015cd45