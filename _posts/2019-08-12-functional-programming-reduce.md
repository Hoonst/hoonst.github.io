---
layout: post
title: "Reduce는 무엇인가??""
categories: [clojure]
comments: true

---
# Reduce 란 무엇인가?
Clojure를 배워가면서 이상하게 함수형 프로그램을 명령형으로 짜는 경향이 있었다. 이는 함수형을 배울 이유를 없애버리는 아주 못된 습관이라고 생각되는 바이며, 이를 해결하기 위해서는 map, filter, reduce를 사용하는 습관을 들여야 한다고 한다. 이 세가지를 함수형 프로그래밍에서 많이 사용하는 이유는 간단했다. <!--more-->

```
무슨 프로그램을 사용하여 코딩을 하든, 결국 추상화 시켜본다면 하는 짓거리는 다 map, filter, reduce이다.
```
내가 지어낸 말이 아니라 정말 그런 정신을 기초로 사용한다. 생각해보면
* map: sequence 하나하나에 개별적인 function을 적용하는 것
* filter: 말 그대로 필터링 해서 내가 원하는 것만 가져오는 것
* reduce: sequence들을 종합해서 하나로 만드는 것

이기 때문에 우리가 행하고자 하는 모든 코딩은 map, filter, reduce 미만 잡이라는 공식이 어느정도는 성립한다.
그 중에 Reduce를 먼저 살펴보고자 한다.

본 포스트를 위해
* https://purelyfunctional.tv/article/annotated-clojure-core-reduce
* https://purelyfunctional.tv/article/a-reduce-example-explained
의 내용을 많이 번역해왔음을 알립니다. 한국어는 함수형의 reduce 설명은 많은데 Clojure에 해당하는 포스트는 적어보였다.

### 기본 개념
Clojure에서는 정말 Reduce를 많이 사용한다. 그리고 이에 대하여 상당히 심오하고 mysterious한 function이라고 여기지만 사실 deep한건 맞는데 mysterious하지는 않다. 정말 단순하게 생각해보자

```clojure
;하나의 컵에 콩을 넣고, 사과를 넣고, 친구를 넣으면 어떻게 될까?
;[콩, 사과, 친구] 가 될 것이다. Reduce는 끝!
(reduce + 0 
	[1 2 3 4 5])
 ;==> 15
```
reduce를 사용하면 initial value를 사용해도 안해도 되는데, 위의 예시에서는 0이 initial value인 것이다. 만약 initial value를 설정하지 않는 다면 sequence의 첫 인자가 initial value가 된다. 
즉, 왼손에 0을 놓고 바닥에 [1 2 3 4 5] 바구니가 있다고 생각을 하면 오른속으로 1을 주워 왼손에 더한다. 그럼 왼손이 1이 되고 오른손은 다시 바구니의 2를 꺼내 왼손의 1에 더한다. 이런식의 반복을 통해 바구니의 숫자들이 다 할 때까지 계산하는 것이 목표이다.

### 타입 변환을 위한 reduce

reduce는 데이터 타입을 바꿔줄 때도 사용할 수 있다.
```clojure
(reduce conj #{}
	[1 2 3 4 5])
;==> {1 4 2 3 5} set는 순서가 뒤죽박죽이라 순차대로 될 것이라는 희망을 버려야 한다.
```
위에서는 conj 자리에 + 함수가 존재해서 왼손에 오른손의 무언가를 계속 더해서 오른손이 deplete될 때까지 조진다의 느낌이었다.
하지면 이번에는 오른손이 depleted 될때까지 set에 conj해 나가는 활동을 이어나간다.

###좀 더 복잡한 것으로 넘어가보자
```clojure
(reduce (fn [a b]
          (if (> a b)
            a
            b))
        0
        [1 2 3 4 5])
```
결과가 어떻게 될 것이라고 생각되는가? 나는 이 syntax가 처음에는 너무 익숙하지 않아 혼났으니 이해했다면 똑똒이가 틀림없다. 답은 5이다.

#### 코드 설명
1. 기본적으로 reduce는 두개의 인자를 받으니 익명함수 fn의 파라미터는 [a b]

2. 만약 a가 b보다 크면 a의 자리는 그대로이며 아니면 굴러온 b가 박힌 a를 몰아낸다.

3. 이를 a = 0 그리고 도전자 [1 2 3 4 5]라고 생각하고 순차적으로 진행한다.

다시 좀 풀어서 말한다면, (위의 왼손 오른손 비유로 해보자면, 왼손에 0을 두고 바구니 [1 2 3 4 5] 에서 오른손을 꺼내 왼손과 계속 비교를 한다. True인 value를 왼손에 놓는 것을 목표로 계속 반복한다.

### 그럼 평균에 대해서도 해볼까?

```clojure
(reduce (fn [[n d] b]
          [(+ n b)
           (inc d)])
        [0 0]
        [1 2 3 4 5])
        
;fn을 아래와 같이도 쓸 수 있는데 추가 학습을 원한다면 살펴보고 아니면 pass
(fn [avg b]
  (let [n (first avg)
        d (second avg)]
    [(+ n b) (inc d)]))
```
이 reduce의 결과는 어떻게 될까? 

#### 코드 설명
1. 익명함수 fn은 [n d]와 b를 인자로 받는다 - reduce에 대한 익명함수이므로 파라미터가 두개여아 하는 것은 당연하다. vector형과 숫자로 받은 것이다.

2. n과 b를 더한 것 그리고 d를 하나 증가 시킨 결과에 대하여 다시 동일한 연산을 한다.

3. 즉 최초 n = 0/ d = 0/ b = 1 인 상황을 예로 들어보자면 첫 번째 결과는 [(+ 0 1) (inc 0)] 이 되어 [1 1]이 된다.

4. 이를 반복하면 결과는 [15 5]

### General Pattern of Reduce
```clojure
(reduce (fn [left right]
          (dosomething left right))
        starting-value
        collection)
```

