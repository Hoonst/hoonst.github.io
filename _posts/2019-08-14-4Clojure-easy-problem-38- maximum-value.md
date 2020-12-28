---
layout: post
title: "(4Clojure) Easy_Problem_38_maximum_value"
categories: [clojure]
comments: true

---
# Maximum value

Difficulty:	Easy
Topics:	core-functions

Write a function which takes a variable number of parameters and returns the maximum value.
<!--more-->

(= (__ 1 8 3 4) 8)

(= (__ 30 20) 30)

(= (__ 45 67 11) 67)

이번에는 숫자들 중 가장 큰 값을 구해내는 함수를 만드는 것이 목적이다. Special Restrictions는 당연히 max와 max-key이다.
먼저 이 문제를 풀기 전에 살짝 당황했다. 왜냐면 여태까지 푼 문제는 대부분 파라미터를 하나만 받았기 때문이다. ? 그게 무슨 말이냐 우리는 이전에도 많은 숫자가 포함된 Sequence를 받지 않았었냐 라고 물어볼 수 있겠지만, 사실 하나의 sequence안에 여러 값들이 존재했던 것이지, 많은 파라미터 자체를 받아본 적은 없다.

따라서 이번에는 다중 파라미터 받는 법 및 Clojure에서 함수를 만들때 파라미터 받는 경우의 수를 설정하는 법을 알아보겠다.

### 먼저 함수를 어떻게 만드냐?
clojure의 syntax는 뭔가 너무 쉽게 만드려고 하다보니 오히려 더 헷갈리가 만들어 버리는 부분이 가끔 존재한다. 함수를 만드는 과정이 대표적인 예시이다. 여태까지는 4clojure의 문제를 풀기 위해 계속 익명함수 (fn)만 만들다 보니 그럼 파이썬에서의 def와 같이 계속 call해서 사용할 수 있는 함수는 만들지 못하는 것인가 라고 오해할 수도 있겠다. 

### def
Clojure에도 def가 존재한다. 하지만 이 def는 함수 지정이 아니라 변수 지정이다. 간단하게 예시를 봐보자
```clojure
(def a 10)

(def hoonst "Super Sexy")
(def 변수이름 value)

hoonst
=> "Super Sexy"
```
위와 같이 hoonst(작성자 id)에 def를 사용하여 "Super Sexy"를 얹었다. 이제 hoonst라는 변수는 "Super Sexy"라는 값을 갖고 있게 된다.

### defn
defn을 통해서 우리는 함수를 만들 수 있다. 간단한 더하기 함수를 만들어보겠다
```clojure
(defn my-plus [a b] (+ a b))
(defn 함수이름 [파라미터] (실제 function)) 순으로 작성

(my-plus 10 20)
=> 30
```
짠~ 쉽게 함수를 만들어보았다. 하지만 이것은 너무 기초이니 만드는 것은 여기까지 하고 가정을 통해 심화학습을 해보겠다.

### 가정 1
> 파라미터가 있을 수도 없을 수도 있다면??	

아니 이게 무슨 말이요... 함수에 파라미터가 없다니? 하지만 실제로 function은 파라미터를 받지 않고 그냥 call하면 자동으로 지정된 값을 return하는 경우가 있긴 하다. 근데 파라미터가 없으면 그냥 차라리 해당 함수의 return 값을 변수에 담는게 더 좋기 때문에 딱히 쓰는 경우는 적다.

한 함수가 필요로 하는 파라미터의 수를 우리는 "arity" 라고 한다. 만약 특정한 갯수의 인자만 받는다면 single-arity이며, 여러 경우의 수를 갖는 다면 multi-arity가 되시겠다. single-arity는 위에서도 설명했기에 multi인 경우만 봐보겠다.

```clojure
((defn punch
  ([name punch-target]
   (str "I will beat the shit out of " name " on the " punch-target))
  ([name]
   (punch name "vital spot")))
```
clojure는 (개인적인 견해로) 재귀를 좀 좋아하는 것 같다. multi-arity인 경우만 봐도 이렇게 짜는 예시가 많이 나와있는데, 만약 두 파라미터가 모두 있다면 온전한 결과를, 파라미터가 모자라면 다시 부족한 부분을 채워서 오리지날 function이 돌아가게 한다.

### 가정 2
> 애초에 파라미터 개수를 예측할 수 없다면?	
파라미터는 하나일 수도 두개일 수도 그 이상일 수도 있다. 그에 맞추어서 위의 punch같이 일일이 작성하는 것은 매우 비효율적이다. 이번 가정에는 &를 사용해서 문제를 해결해보자

```clojure
(defn kick [name & rest] (print "I will kick " name " and " rest "too"))

(kick "Hoon" "창현" "은솔")
I will kick  Hoon  and  (창현 은솔) too=> nil)
```
느낌이 혹시 오는가? &를 사용하게 되면 하나는 왼쪽에 나머지는 rest에 담기게 된다. 이것이 아직 유용하다고 느끼지 못할 수 있으나 알고 있으면 쓸 일이 반드시 존재한다.

만약 이렇게 설정할 일도 없고, 그리고 왠만하면 가정을 통해 파라미터를 받는게 아니라 한번에 dump해놓고 사용하고 싶다면?

오늘의 max value찾기에서 이 사용법을 알아보자

```clojure
((fn [& args]
   (reduce (fn [x y]
             (if (> x y)
               x
               y)) 0 args)) 2 8 10 1 5)
```
### 코드 설명
1. 익명함수 fn은 args라는 argument들을 받고 그것을 하나의 sequence로 받는다.
2. reduce를 사용해 initial value는 0 / args는 sequence로 설정하고 args를 하나씩 돌아서면서 initial value와 비교를 통해 교체한다.

즉 여러 value를 sequence에 dump해서 마치 sequence를 input으로 하는 함수를 만들고 싶다면 [& args] 를 사용하는 것이 좋다. 사실 저 args라는 syntax는 정해진 것이 아니라 마음대로 바꿔도 된다.

그럼 혹시 다른 형님들의 정답은 어떻게 이루어졌는지 살펴봐보자.

### 1067's solution
```clojure
(fn [& params] (reduce (fn [a b] (if (< a b) b a)) 0 params))
```
짜잔~	
1067 형님의 정답은 나의 것과 완벽하게 일치한다.

### maximental's solution:
```clojure
#(last (sort %&))
```

뭔가 많은 것을 설명하려고 했는데 은근히 가독성이 떨어지게 작성한거 같아 마음이 착잡합니다...

![see-ya]({{ "/assets/img/pexels/see_you_next_time.jpg" | relative_url}})