# ARIMA Model

본 포스트는 고려대학교 산업공학과 김성범 교수님의 **예측모델** 수업을 토대로 작성한 것을 밝히는 바입니다.

해당 내용은 ARIMA Model에 다가가기 위하여 기본적인 확률 통계에 대한 Notation을 정리한 것입니다. 



$[x_1, x_2, x_3 ... x_t]$ : a sequence of random variables ~ 시간에 의한 Sequence

* $F_X(x) = P(X \leq x)$ : Cumulative Density Function (CDF)

* $E(x) = \mu x$
* $V(x) = E[(x-\mu x)^2] = \sigma x^2 $
* $x_1, x_2$ 의 공분산: $Cov(x_1, x_2) = E[(x_1 - \mu_1)(x_2 - \mu_2)] = \sigma(x_1x_2)$ 
* $x_1, x_1$ 의 공분산, 즉 자기 자신의 분산: $Cov(x_1, x_1) = V(x_1) = \sigma x_1^2$ 

* $x_1, x_2$의 상관계수: $Corr(X_1, X_2) = \frac {Cov(x_1, x_2)}{\sqrt{V(x_1)V(x_2)}} = \frac{\sigma(x_1x_2)}{\sqrt{\sigma x_1^2\sigma x_2 ^2}} = \frac{\sigma(x_1x_2)}{\sigma x_1 \sigma x_2}$

If $X, Y$ is Independent

* 서로 독립인 확률변수의 곱의 기댓값은, 각 확률변수의 기댓값의 곱과 같다: $E(X \cdot Y) = E[X] \cdot E[Y] $

* $Cov(X, Y) = 0$

* $Cov(X+2, Y) = Cov(X, Y) + Cov(2, Y) = Cov(X, Y) +E[2Y] - E[2]E[Y] = Cov(X, Y) + 2E[Y] - 2E[Y] = Cov(X, Y)$

* $Cov(X, Y) = Cov(Y, X)$

* $Cov(aX) = aCov(X, Y)$ ??????

  

## ARIMA Model에서 많이 사용하는 개념을 정리

**Autocovariance: N 시점 전과의 데이터와 Covariance**

**[Covariance]**

* 같은 확률변수지만 $h$  시점 차이의 공분산: $Cov(X_t, X_{t+h}) = \gamma_x(h)$

  $\gamma$의 특징

  1. $\gamma_x(0) \rightarrow$ $h$시점 차이를 나타내는 식의 $h$ 가 0이므로 시점 차이가 없고, 이는 해당 시점의 공분산, 즉 분산을 뜻한다.
     $\Rightarrow Cov(x_t, x_t) = V(x_t) = \sigma x_t^2$

  2. $\gamma_x(-h) \rightarrow Cov(x_t, x_{t-h})$

     $ = Cov(x_{t-h}, x_t) = Cov(x_{t-h}, x_{(t-h) + h})$

     $ =\gamma_x(h)$

     $\Rightarrow \gamma_x(h) = \gamma_X(-h)$, 즉 모든 $h$에 대하여 $\gamma$는 *Symmetric*하다

**[Correlation]**

* $\rho_x(h) = \frac{Cov(x_t, x_{t+h})}{\sqrt{V(x_t) \cdot V(x_{t+h})}} = \frac{\gamma_x(h)}{\sqrt{\gamma_x(0) \cdot \gamma_x(0)} = \gamma_x(0)}$

  $V(x_t) , V(x_{t+h})$가 $\gamma_x(0)$로 모두 변하는 것이 조금 의아할 수 있다. 하지만 현재 우리가 살펴보고 있는 것은 'Autocorrelation'이므로,
  하나의 시점 내의 분산은 시점의 변화가 없어 $\gamma_x(h)$의 $h$부분이 0이다.

* $\rho_x(0) = \frac{\gamma_x(0)}{\gamma_x(0)} = 1 \rightarrow Corr(x_t, x_t)$
* $\rho_x(-h) = \rho_x(h)$ , for all $h$
* $-1 \leq \rho_x(h) \leq 1$



**[White Noise] 백색잡음, 백색 Noise**

$a_t, WN(a_t) \rightarrow$ Sequence of Random Variable

* $E(a_t) = 0, \forall t$
* $V(a_t) = \sigma a^2. \forall t \rightarrow$ $V$는 $t$ 에 대한 식이 아니라, $a_t$에 대한 식이므로, 모든 시간대에 대하여 분산이 일정하다.
* $Corr(a_t, a_s) = 0 / t\neq s$: 상관이 0이라는 뜻이므로, 독립임을 나타낸다.
* $\gamma_a(h) = Cov(a_t, a_{t+h}) = \begin{cases}
  \sigma a^2 & h = 0 \\
  0 & h \neq 0
  \end{cases}$     $h$ = 0 은 자기 자신에 대한 공분산, 즉 분산이 되며, $h \neq 0$은 공분산이 0이 된다.

* $\rho_a(h)  = \begin{cases}
  1 & h = 0 \\
  0 & h \neq 0
  \end{cases}$

  

**[ARIMA]**

* $E(X_t) = \mu$

* $Var(X_t) = \sigma X_t^2$

  $\Rightarrow$ 시간에 따라 일관되게 Constant한 확률분포를 가짐

  $\Rightarrow$ Stationary Time Series

**Example)**

$Z_t  = \beta_0 + \beta_1 t + X_t$ 이며 $X_t$가 Stationary라 하자.

그렇다면 $Z_t$가 Stationary 하지는 않지만, $\Delta Z_t = Z_t - Z_{t-1}$는 Stationary 함을 보여라.

> Stationary: $E(X), V(X)$ 가 $t$에 상관없이 일정해야 한다.



* $E(Z_t) = E(\beta_0 +\beta_1 t + X_t) = \beta_0 + \beta_1 t + E(X_t)$

  $= \beta_0 + \beta_1 t + \mu$

  즉, $Z_t$의 기댓값 또는 평균이 $t$에 따라 달라질 것이므로, 평균이 일정해야 만족하는 정상성을 달성할 수 없다.

* $E(Z_t - Z_{t-1}) = E((\beta_0 + \beta_1 t + X_t) - (\beta_0 + \beta_1(t-1)+X_{t-1}))$

  $ =\beta_1 + E(X_t) - E(X_{t-1})$

  $X_t$와 $X_{t-1}$는 정상성을 만족하는 시계열 데이터이기에 평균이 모두 같은 $\mu$ , 즉 둘이 상쇄되어 사라지고 $\beta_1$만 남게 되어 일정한 값이 된다. 

* $Cov(\Delta Z_t, \Delta Z_{t-h}) = Cov(Z_t - Z_{t-1}, Z_{t-h} -Z_{t-(h+1)})$

   = $Cov(Z_t, Z_{t-h}) - Cov(Z_t, Z_{t-h-1}) - Cov(Z_{t-1}, Z_{t-h}) + Cov(Z_{t-1}, Z_{t-h-1})$

   = $\gamma_Z(h) - \gamma_Z(h+1) - \gamma_Z(h-1) + \gamma_z(h) $

   =$2\gamma_Z(h) - \gamma_Z(h+1) - \gamma_Z(h-1)$

해당 식은 $t$에 대한 식이 아니므로, 시간에 따라 변하지 않는 일정한 값이 될 것이다.



이로써 $Z_t$는 정상성을 만족하지 않지만, $\Delta Z_t$는 만족하는 것을 살펴보았다.

사실 $\Delta Z_t$는 식의 의미를 생각해보면 차분임을 알 수 있으며, 비정상성 데이터에 대한 차분이 정말 정상성을 만드는 지에 대한 증명이라 할 수 있다.