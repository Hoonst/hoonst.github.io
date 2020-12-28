---
layout: post
title: "(4Clojure) Easy_Problem_33_replicate_sequence"
categories: [clojure]
comments: true

---
# Replicate a Sequence
Write a function which replicates each element of a sequence a variable number of times. <!--more-->

```clojure
(= (__ [1 2 3] 2) '(1 1 2 2 3 3))


(= (__ [:a :b] 4) '(:a :a :a :a :b :b :b :b))


(= (__ [4 5 6] 1) '(4 5 6))


(= (__ [[1 2] [3 4]] 2) '([1 2] [1 2] [3 4] [3 4]))


(= (__ [44 33] 2) [44 44 33 33])
```

Sequence를 받고 그것을 주어진 횟수대로 반복하는 함수를 만들어야한다.
처음에는 전체 Sequence를 받고 그것을 주어진 횟수대로 반복하려 했으나 이는 옳지 않았다. 
이 말은 즉슨,
```clojure
(reduce into [] (repeat 2 [1 2 3]))
```
### 코드 설명
1. reduce 를 통해 [] 안에 
2. [1 2 3]의 repeat 2번한 결과를 차례로 집어 넣는다.

하지만 이것의 결과는 [1 2 3 1 2 3] 으로써 우리가 원하는 결과인 [1 1 2 2 3 3]은 아니다. 
즉 개별 인수들에 대한 접근이 일일이 필요한 것이다.

이를 위해 map을 통해 개별 인자에 대한 계산을 적용해야 한다.

이를 위하여 처음에 시도한 방법:
```clojure
((fn [lists repeat-time]
   (map (fn [lists]
          (repeat repeat-time lists))
        )) '(1 2 3) 2)
==> #object[clojure.core$map$fn__5847 0x17b6c9b4 "clojure.core$map$fn__5847@17b6c9b4"]
```
원하는 결과는 나오지 않고 계속 이상한 object만 return한다. 
이유를 찬찬히 생각해봤을 때 map에 이유가 있음을 알게 되었다.
현재의 map (function)을 살펴보았을 때 함수 다음에 적용해야 하는 대상이 없다. fn [lists] 다음에 대상이 없기 때문에 아무것도 실행하지 못하고 넘어가는가 싶다. 

즉, map (function)에 대해서 다시 상기해보자면 sequence의 인자들 하나하나씩에 대하여 주어진 function을 적용하는 것이다. 따라서 이 map에 사용되는 function을 위한 인자를 집어 넣어줘야 한다. 그렇게 작성하는 방법은 다음과 같다용.

```clojure
((fn [lists repeat-time]
   (mapcat #(repeat repeat-time %) lists)) [1 2 3 4] 2)
;==> (1 1 2 2 3 3 4 4)
```
익명함수를 작성하는 방법은 대표적으로 두가지가 있는데 하나는 fn()의 방식으로 작성하는 것이며, 두번째는 #()의 방식으로 작성하는 것이다. 그리고 % 자리에는 input이 들어가게 된다. 
또한 처음에는 map을 적용한 후에 concat이나 conj / into를 통한 결합을 하려 했는데 놀랍게도 이것을 염두해둔 function이 따로 있었으니 **mapcat**이다!

mapcat은 map을 적용한 뒤에 결과들을 하나로 seq로 만들어주는 역할을 한다.

그렇다면 이번에도 어김없이 다른 행님들의 답안을 봐보도록 하자.
그런데 의외로 형님들의 답안이 크게 나의 것과 다르지 않아 올려놓기만 하고 넘어가도록 하겠다.

### 1067 형님
```clojure
(fn [xs n] 
	(reduce concat (map #(repeat n %) xs)))
```

### maximental 형님
```clojure
(fn [x n] (mapcat #(repeat n %) x))
```

*이상 전달 끝*

![lot]({{ "/assets/img/pexels/lot.jpeg" | relative_url}})

이 세상 최고의 만화, 네이버 웹툰 [덴마]로부터...
하루하루 문제를 풀어나가면서 덴마 짤을 사용할까 싶다

   