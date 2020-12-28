---
layout: article
title: "Bagging Predictors - Silk Purse out of a sow's ear"
tags: paper_review
mathjax: true
article_header:
  type: cover
  image:
    src: 
---

# Bagging Predictors
Author: LEO BREIMAN(1996)
*Statistics Department, University of California, Berkely, CA 94720*
<!--more-->
___________________

신기하게도 Bagging에 대해서 자세하게 알아보고 싶고, 원 논문을 탐닉함에 앞서 다른 사람들은 본 논문에 대한 리뷰를 어떻게 했을까 싶어서 검색을 해본 결과, 정리나 리뷰를 해놓은 사람이 한국어, 영어 모두 존재하지 않았다. 그 이유가 궁금하기도 하다.

### 주의사항:
- 본 논문은 Notation이 꽤 헷갈린다.
- 사실 논문의 Notation은 언제나 어렵지만, 너무 지나치게 왈가왈부하는 것 같지만 모조리 정리하면서 넘어가보고 싶다.


**Abastraction**:
- **Bagging Predictors**란 여러 종류의 predictor를 만든 뒤, 그것들을 종합하여 만든 aggregated predictor를 칭한다.  
- **Aggregation** 방법:
  * Regression: Predictor들을 평균
  * Classification: 다수결 투표

- 여러 버전의 모델들은, 데이터의 **Bootstrap** 결과들을 Learning Set으로 사용하여 훈련을 시킴으로써 생성된다.  
- 여러 데이터들에 대하여 Bagging을 적용해본 결과,
  * Classification Tree
  * Regression Tree
  * Subset Selection in Linear Regression

에서 큰 성능 향상을 보였다.
- 성능 향상의 정수는, 예측 기법의 불안정성이다.
  * learning set을 변경하는 것이 predictor에 영향을 크게 준다면, baggins의 성능은 증가할 것이다.


용어 정리:  
* Predictors: 목적 함수
* $L$: Learning Sets = {${(y_n, x_n), n=1,...,N}$}
  $L_k$: $L$과 같은 분포에서 뽑인 Learning Sets
* $\varphi$: Predictor

Introduction:
Bagging의 일반적인 절차는 다음과 같다.
하나의 Learning Set은 {${(y_n, x_n), n=1,...,N}$}와 같이 구성되어있다. 이때, $y_n$은 Numeric일수도 class label일수도 있다. 이 때 $\varphi(x, L)$은 $x$를 Predictor에 넣어 y를 산출하는 과정이다.

> $y$ = $\varphi(x, L)$

여기에 $L_k$가 등장하게 되고, $L$과 같은 분포에서 뽑인 Learning Sets를 뜻한다. 즉, $k$의 수에 따라 Learning Sets의 갯수가 다르며, Bagging의 큰 목적은 여러 개의 $L_k$로 훈련시킨 Predictors의 결과가 단일의 Predictor보다 좋아야 하는 것이다.

* $y$가 Numeric이라면 **단일의 $\varphi(x, L)$를 다중의 $\varphi_A(x, L_k)$ 대체하는 것이 목적** 이다.
> $\varphi_A(x, L_k)$의 A 첨자는 Aggregation을 뜻한다.
즉 , $\varphi_A(x, L_k) = E_L\varphi(x, L)$이다.

* $y$가 class $j \ {1,...,J}$라면, $\varphi(x, L_k)$를 aggregate하는 것이 bagging 방법 중의 하나이다. '방법중의 하나' 라는 말은 더 읽어보다보면 이해하게 될 것이다.

| Notation| Meaning |
| ----------------- | ---------------- |
| $E_L$ | $L$에 대한 기댓값 |
| $\varphi_A$ | $\varphi$의 aggregation |

하지만 뭔가 이상하다. 위에서 말한 방법이 다소 형이상학적이다. 왜냐면 $L_k$를 L과 같은 분포에서 뽑는다고 하는데 실상 우리는 해당 분포를 모른다. 따라서, 분포에서 $L_k$를 뽑아내는 행위를 모방하는 절차로서, Bootstrap을 사용한다. 랜덤하게, 복원추출을 진행하는 Bootstrap을 진행함으로써 $L_k$ 대신 $L^{(B)}$를 취하게 된다.

* $L > L^{(B)} > \varphi(x,L^{(B)})$  L에서 L bootstrapped을 추출 후, predictor에 전달
* $\varphi_B(x) = av_B\varphi(x, L^{(B)})$ Bootstrap predictor의 결과는 **평균** 이다

![](/assets/BaggingPredictors/2020-09-03-Bagging-Predictor-747cd5c9.png)

##### Bagging의 critical factor
Bagging의 핵심은 바로 '불안정성'이다. 즉, predictor인 $\varphi$가 얼마나 불안정한지에 따라 Bagging의 성능이 올라간다.

만일 데이터 셋 $L$의 변화가 $\varphi$의 **작은 변화** 만을 가져온다면 $\varphi_B$는 $\varphi$와  매우 유사할 것이다. 즉, $L$의 작은 변화가 predictor $\varphi$의 큰 변화를 가져올 수 있다면, Bagging의 성능은 향상될 것이다.
Unstability에 대해서 Predictor를 분류해보자면,
* Neural Nets, Classification and Regression Trees(CART) Subset Selection in Linear regression 정도가 불안정하고
* KNN은 Stable 하다고 한다.


이에 대해서 개인적인 생각을 해보자면, 기존 $\varphi$와 변경된 $\varphi_B$가 유사하다면, 굳이 Bootstrap을 하여 보고자 한 새로운 관점이 사라질 것이므로, '새로운 것'이 발견되지 않는 것이다. 하지만 Bootstrap으로 만들어진 새로운 데이터 셋이 새로운 현상 또는 결과를 발견한다면 미처 발견하지 못한 정보를 얻게 되어 전체적으로 성능이 증가하지 않나 싶다.

### Why Bagging Works?

간단하게 생각해보면, Bagging의 성능이 증가하는 것은 당연하다. 한번의 계산보다 여러 번의 계산을 진행하니 말이다. 하지만 진정으로 왜 성능이 보장되는 지에 대해서 아직 다루지 않았다. 알아보도록 하자.

본 논문에서는 성능의 보장을 Regression, Classification, Subset Selection을 예시로 하나씩 설명한다. 해당 파트는 수식이 매우 많이 나오고, 상당히 난해해서...각오하길 바란다.

#### Regression

$(y,x)$를 $L$의 원소라고 하고, $L$은 $P$ 분포로부터 왔다고 가정해보자.
또한 y는 numerical, predictor는 $\phi(x, L)$이다.
> 본 논문에서는 이상하게 $\varphi$와 $\phi$가 혼용되어 사용된다. 나는 이 부분에 대하여 설명을 찾고자 했지만, 이상하게 나타나질 않았다. 하지만 행간에서 의미를 추론해보자면, $\varphi$는 모집단, 즉 Learning Set의 원래 분포를 사용하는 함수, $\phi$는 단순히 Learning Set의 분포를 나타내는 것이 아닐까 싶다.

> $\phi_A(x)$ = $E_L\phi(x, L)$

$x$와 $y$를 각기 fixed input과 output으로 생각한 뒤, MSE(Mean Squared Error)를 구해보자.

> $E_L(y-\phi(x, L))^2 = y^2 - 2yE_L\phi(x, L) + E_L\phi^2(x, L)$

여기서 우리는 Jensen's inequality를 우변의 세번째 항 $E_L\phi^2(x, L)$에 적용할 것이다.
> Jensen's inequality: 기댓값의 함수와 함수의 기댓값 사이에 성립하는 부등식이다.
이때 함수의 꼴이
* 볼록 함수(위로 볼록 e.g. $-x^2$)이면 $f(E(X)) \leq E(f(X))$이며,
* 오목함수 (위로 볼록 e.g. $x^2$)이면 $f(E(X)) \geq E(f(X))$이다.

우변의 세번째 항 $E_L\phi^2(x, L)$는 형태를 살펴보면 $f(x)$가 $x^2$이고, x가 $\phi(x, L)$인 꼴이다.
따라서 이는 오목함수의 inequality를 따르고 $f(E(X)) \geq E(f(X))$를 따르게 된다.
즉, 우변 세번째 항을 $E_L\phi^2(x, L)$에서 $(E_L\phi(x, L))^2$로 바꾸게 된다면, 우변 전체는

> $y^2 - 2yE_L\phi(x, L) + E_L\phi^2(x, L)$에서

> $y^2 - 2yE_L\phi(x, L) + (E_L\phi(x, L))^2$로 바뀌게 되고 원래의 함수와 대소관계를 갖게 될 것이다.

 > $E_L(y-\phi(x, L))^2 \geq y^2 - 2yE_L\phi(x, L) + (E_L\phi(x, L))^2 = (y-E_L\phi(x, L))^2 = (y-\phi_A(x, L))^2$

 이 부등식은 $[E_L\varphi(x, L)]^2 \leq  [E_L\varphi^2(x, L)]$ 의 차이가 커질 수록 더 커지게 된다.



### Classification
이전에는 분류문제를 Bagging으로 접근할 때, 다수결 voting으로 진행한다고 했엇다.
예를 들어, 여러 predictor에서 1,2,3을 classification을 한다고 했을 때, 1 label이 다른 label보다 많이 예측되었을 때, 다수결로 해당 bagging의 결과는 1이라고 결론을 내는 것이다. 하지만 본 단락에서는 이를 확률적으로 접근하여 문제를 헷갈리게 만든다. 이를 명확하게 하는 시간을 가져보자.

분류 문제에선 $\varphi(x, L)$는 $j\in${1,...,J} 이며
> $Q(j|x) = P(\varphi(x, L) = j)$ 이다.

Q와 P는 상당히 헷갈리고 분간이 잘 안되는데 $Q(j|x)$, $P(j|x)$를 다음과 같이 표현한다.

* $Q(j|x)$: 이는 기본적으로
> $Q(j|x)$ = $P(\phi(x, L) = j)$
> Over many independent replicates of the learning set $L$,
$\phi$ predicts class label j at input x with relative frequency $Q(j|x)$
> $P(j|x)$: x


![](/assets/BaggingPredictors/2020-09-03-Bagging-Predictor-8fd472ae.png){:.border.rounded}

### Using the Learning Set as a Test Set

Bagging tree에서 $L_B$는 $L$의 분포인 $P_L$에서 표집된다. 그리고 $L_B$를 통해 T tree가 생성된다. CART 알고리즘의 최적의 subtree가 CV나 test로 인해 선정된다.

그렇다면 학습을 다 마친 뒤에 이를 Test하기 위한 Test set은 어떻게 산출해낼까?
기존의 train, validate, test는 각자의 영역이 존재하는데, Bagging을 활용하게 되면 이 경계가 존재하지 않는다. Learning Set $L$에서 train validate test 모두 산출되기 때문이다.


### Simulation Structure


### How many Bootstrap replicates are enough?

본 논문에선 classification에선 50, regression에선 25개의 bootstrap이 사용되었다. 저자의 말로는
어떤 근거가 있어서라기 보단, 그냥 그럴 것 같았고, regression은 Bootstrap을 적게, Classification에서는 분류 class가 많을 수록 많이 설정하는 것이 좋다고 한다.

| No.Bootstrap Replicates| Missclassification Rate |
| :-----------------: | :----------------: |
| 0 (unbagged) | 29.1 |
| 10 | 21.8 |
| 25 | 19.4 |
| 50 | 19.3 |
| 100 | 19.3 |

0개에서 10개 / 10개에서 25개 ...로 갈 수록 점점 오분류율이 감소한다. 하지만 그 감소폭이 거의 10개에서 큰 발전이 없는 것 같으므로, 10개정도의 Bootstrap Replicates라면 충분하다는 말이다. 물론 더 해도 되지만... 시간 용량 낭비의 느낌이다.

### 6.3 How Big Should be Bootstrap Learning Set Be?

Bootstrap을 사용할때 $L^{(B)}$의 크기를 원래 $L$의 크기와 같게 해서 실험을 진행했다. Bootstrap이라는 복원 추출을 사용했으니 각 Bootstrap Learning Set, $L^{(B)}$는 2,3...개의 중복이 발생할 수 있으며, 보통 .37 정도의 데이터가 표집이 안된 상태라고 한다. 이 논문과 연관이 있는 논문을 읽은 사람이 이 수치는 꽤 영향력있는 정도라고 평가하여, "만약 Bootstrap을 두배로 한다면 성능이 증가하지 않을까" 라는 제안을 해서 실험해보았더니, $e^2$ = .14 정도의 loss를 보였다고 한다. 하지만 이것이 성능을 증가시키진 못했다.

### Conclusion
Bagging은 여러 weak한 prediction을 aggregate하여 단일한 prediction의 성능을 뛰어넘고자 하는 노력의 알고리즘이다.
특히 해당 prediction에 사용되는 알고리즘이 불안정성을 띌 때 효과가 더 크다. Bagging은 기존의 방법을 향상시키는 매우 간단한 알고리즘인데 그 이유는 단순히 '여러번' 하고 다 더해버리면 되기 때문이다. 하지만 tree의 경우, 원래는 간단하고 해석이 용이했는데 bagging으로 인해 해당 장점을 잃을 수 있긴 하다. 하지만 성능은 보장된다!
