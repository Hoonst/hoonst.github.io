---
layout: post
title: "(4Clojure) Easy_Problem_27_Palindrome"
categories: [clojure]
comments: true

---
# (4Clojure-Easy) 27번 Palindrome Detector 문제

Fibonacci와 마찬가지로 이놈의 Palindrome 역시 기초 좀 떼고 기지개 좀 피려고 할 때 나타나서, 
**"아직 너는 약한 존재다"** 라고 방해를 하는 존재라고 생각이 든다. <!--more-->

### Palindrome 이란?
Palindrome이란 앞으로 읽어도 거꾸로 읽어도 똑같은 단어를 말한다. 한국말로는 회문이라고 하는데 영어로 예를 들자면, "Eye", "Racecar", "Mom" 같은 것들이 있으며 한국어로는 기러기, 이효리 쯤 아닐까 싶다.
이 Palindrome을 판단하려면 앞글자와 뒷글자가 모두 같아야 한다.

```clojure
(false? (__ '(1 2 3 4 5)))

(true? (__ "racecar"))

(true? (__ [:foo :bar :foo]))

(true? (__ '(1 1 3 3 1 1)))

(false? (__ '(:a :b :c)))
```

나는 이 문제를 풀기 위하여 다음의 익명함수를 작성해보았다.
```clojure
(fn [palin]
   (loop [check-palin palin]
     (if (empty? check-palin)
       true
       (if (= (first check-palin) (last check-palin))
         (recur
           (drop-last (rest check-palin)))
         false))))
```
### 코드 설명
1. palindrome 판단을 하고 싶은 문자열이나 list를 인자로 받는다.

2. loop-recur를 시작한다. 이때 check-palin은 palin을 그대로 받는다.

3. 만약 check-palin이 empty라면 true로 받고 -- 끝까지 에러가 없다는 뜻

4. empty가 아니라면 다시 if를 사용해 check-palin의 first와 last가 같은지 판단

5. 이 조건이 맞다면 check-palin의 처음 다음의 것만 남기는 rest와 그것의 drop-last를 통해 맨 앞과 맨 뒤를 쳐내며,

6. 만일 틀렸다면 false를 return한다.

아직 clojure 하수로서 작성한 이 코드는 내가봐도 뻘짓이 많이 포함되어 보인다. 그렇다면 우리 고수 형님들이 어떻게 작성했는지 살펴보자.

### 1067' Solution
여윽시 1067 형님, 4Clojure를 뒤집어 놓으셨다...
```clojure
(fn [xs] (= (reverse xs) (reverse (reverse xs))))
```
reverse 함수로 간단하게 해내셨다. 하지만 reverse같은 함수는 프로그래밍을 연습하고자 할 때는 다소 좋은 함수는 아니라고 본다. 이 reverse까지 함수에 포함시키는 과정 또한 공부의 연속이기 때문이다. 

간단하게 읽어보면

### 코드 설명
1. xs를 input으로 받고,
2. xs를 reverse 하고 reverse 한 것 == 사실 그냥 xs
3. 그리고 xs를 reverse한 것이 같은지 비교

그렇다면 우리 maximental 형님것을 봐보자

### maximental's solution
```clojure
#(= (seq %) (reverse %))
```
파이썬 user라면 연상되는 것이 있을 수 있다. 

```python
"I eat %d high octane crap to maintain this shit body." % 3
```
포맷팅을 통해 %d 자리에 3을 넣을 수가 있는 것처럼 위의 Clojure 에서도 % 자리에 위 문제에서 주어진 리스트나 문자열을 집어넣을 수 있다. seq와 같은 경우는 그냥 그대로 return한다고 이해하면 되고 reverse는 말 그대로이다.

