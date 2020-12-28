---
layout: post
title: "(4Clojure) Easy_Problem_26_fibonacci"
categories: [clojure]
comments: true

---
# (4Clojure-Easy) 26번 피보나치 문제
Write a function which returns the first X fibonacci numbers.
피보나치 문제는 기본적으로 프로그래밍 학습에 있어서 가장 대표적인 문제가 아닐까 싶다. 가장 기초를 익히고 그 다음으로 이거 한 번 해봐라 했을 때 못하면 자괴감이 들면서도 자신의 기초에 대하여 한탄하게 해주는 문제이다. 피보나치를 만든 양반은 내 생각에 프로그래밍 기초 학습을 위한 문제로 피보나치 수열을 만들지 않았나 싶기도 하다. (죽었으면 좋겠다고 생각한 찰나, 이미 돌아가셨겠거니 싶다) <!--more-->

```clojure
(= (__ 3) '(1 1 2))

(= (__ 6) '(1 1 2 3 5 8))

(= (__ 8) '(1 1 2 3 5 8 13 21))

```
아직 명령형 프로그래밍에 익숙하고 함수형 사고가 덜 되어서 모든 문제를 다소 for-loop로 풀려고 하는 습관이 강하다. 그래서 loop-recur를 사용하는데 사실 loop-recur는 많이 사용하지 않으며, 살짝 Clojure의 세계에선 filter-map-reduce 미만 잡이라면서 3대장 대우를 해준다. 

하지만 먼저 내가 작성한 loop 문을 살펴봐보자

``` clojure
(fn [fib-input]
   (loop [fib-list [1 1] fib-length fib-input]
     (if (= (count fib-list) fib-length)
       (apply list fib-list)
       (recur
         (conj fib-list (reduce + (take-last 2 fib-list)))
         fib-length
         ))))
```

### 코드 설명

1. fn 익명함수는 fib-input을 입력 파라미터로 받는다.

2. loop를 시작한다. 이때 초깃값 설정을 해주는데 fib-list는 [1 1]의 vector, fib-length는 fib-input을 그대로 받는다.

3. 만약 fib-list가 fib-length 와 같으면 return vector를 list로 바꾸어서 return한다. 4Clojure 문제가 원하는 대로 결과를 주기 위해서

4. if가 false (fib-list가 fib-length와 길이가 같지 않다면) 라면 기존의 fib-list [1 1] 에서 마지막 2개를 꺼내고 더한 다음, 자신의 꼬리에 더한다.

5. 새로운 fib-list와 변하는 과정이 없는 fib-length를 recur 시킨다.

솔직히 함수형 프로그래밍을 배운다는 사람이 이렇게 쓰면 안된다는 느낌을 폴폴 받는다. 이것은 명령형 프로그래밍을 Clojure를 사용해서 작성한 것일 뿐, 함수형 프로그래밍의 정수를 사용한 코드는 아니라고 본다.

따라서 Clojure 좀 쓴다는 형님들의 코드를 살펴봐보자.

#1067 형님's solution
```clojure
(fn fib [n]
  ((fn iter [c res]
     (if (= c n)
        (reverse res)
        (let [next (if (< c 2) 1 (reduce + 0 (take 2 res)))]
        	(iter (inc c) (conj res next)))
     )) 0 ()))
```
솔직히 다른 문제는 모르겠지만 피보나치는 내 함수가 더 보기 깔끔하지 않나 싶다. 하지만 한번 읽어나보자

###코드 설명

1. fib (익명)함수는 n 을 파라미터로 받고

2. iter (익명)함수는 c와 res를 파라미터로 받으며

3. 만약 c랑 n이 같으면 res 를 reverse한다.

4. 아니라면 let 구문 속에서 next라는 놈에 대하여 c가 2보다 작으면 1이라 하고, 그렇지 않다면 res의 앞 두개의 합으로 한다

5. 그렇게 정해진 next 자식을 데려다가 c를 하나 inc 시키고 res와 next를 conj!

6. c랑 res는 각기 0 과 ()이다. 시작해라!!

하 총평은 상당히 복잡해진 느낌. 하지만 이렇게 짜는 것이 함수형이라면 따르리...
여기서 reduce function에 대한 의문이 있다.

![reduce]({{ "/assets/img/pexels/Reduce-function-in-Python.jpeg" | relative_url}})

만약 reduce가 저 위의 것 처럼 초기값이 존재한다면, 
* 그 초기값과의 뒤의 리스트의 값 하나하나씩을 더하는 것인지, 
* 아니면 뒤의 리스트의 값을 순차대로 더하는 것인지?

예를 들어보자면, initial value가 0이라면

```clojure
(reduce + 0 [1 2 3])
```
는 0 + 1의 값을 2랑 더하고 3이랑 더하는 것인지
아니면 0에 차례차례 더해가는 것인지 궁금하다. 사실 이 둘의 결과 값의 차이는 없다고 보고 있지만
initial 값에 누적으로 더하는 것인지, 아니면 initial값이 단순히 [1 2 3]에 포함이 되어서 [0 1 2 3]으로서의 계산인건지 궁금해졌다.

즉, reduce에 대한 포스트는 세밀하게 다시 다뤄보도록 하겠다 --> ()

또한 1067 형님의 코드에는 let이라는 함수가 또 있는데 이는 간단하게 말하면, 
let 뒤에 []를 사용하고 [] 안에 기존에 갖고 있는 변수(immutable이 functional programming의 정수이지만 용어의 편이성을 위하여)에 대해서 변이를 시켜주는 것이다. 주로 destructing에서 사용이 되며 연관 개념은 ()에서 다루겠다.

# maximental's solution
```clojure
(fn [n] 
  (take n
        (map first 
             (iterate (fn [[f s]] [s (+ f s)]) 
                      [1 1]))))
```
개인적으로 우리 막시멘탈 형님의 코드가 더 간결하고 읽기 편하다.

### 코드 설명

1. (익명)함수는 n을 파라미터로 받아

2. n개를 take 해온다 (take는 sequence에서 앞의 n개까지 가져오는 것)

3. 각 인수에 대하여 first를 적용한다, map을 통해

4. 그 대상은 iterate, 즉 무한대로 뻗어나가는 계산인데 

5. (익명)함수에 [1 1]을 초기값으로 받아 앞의 인수를 f, 뒤의 인수를 s로 받고 [s와 f + s] 꼴로 계산을 이어나간다.
