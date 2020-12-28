---
layout: post
title: "(4Clojure) Easy_Problem_34_implement_range"
categories: [clojure]
comments: true

---
# Implement range
 
Difficulty:	Easy	
Topics:	seqs core-functions

Write a function which creates a list of all integers in a given range.
<!--more-->
```clojure
(= (__ 1 4) '(1 2 3))

(= (__ -2 2) '(-2 -1 0 1))

(= (__ 5 8) '(5 6 7))
```
빈 칸에 함수를 넣어 range를 직접 구현하는 것이다. 하지만 4Clojure에서는 가끔 이렇게 기존에 존재하는 함수가 있을 떄는 Special Restriction이라면서 사용하지 못하는 함수를 정해둔다. 이런 경우는 뭘 밴하겠는가?

**Range** 를 밴하겠지요
(혹시 되나 싶어서 그냥 range 넣어봤는데 혼나고 만다)

다행히도 이번 문제는 여러번에 시행 착오 없이 의자에 잠깐 누워서 구상을 했더니 바로 풀렸다.	
드디어 학습의 뽕이 살짝 들어오는 느낌이었다.

![haz_ponder]({{ "/assets/img/pexels/haz_ponder.jpg" | relative_url}})	
(이 돼지인간은 덴마에서 본명 '하즈' 별명으로는 '돼갈량'이라고 불리는 지략가이다. 	
그의 생각을 보다보면 지려버릴 수도 있다)

본격적으로 문제를 풀어보도록 하자.
```clojure
((fn range-generator [start end]
   (take (- end start) (iterate inc start))) -2 2)
   ```
### 코드 설명
1. 익명함수 range-generator (사실 익명이라고 했는데 왜 이름이 있나 궁금할 수도 있는데, 넣을 수도 있음을 보여드리는 바입니다요) 는	
input으로 start, end 두개의 파라미터를 받는다.

2. range 함수는 보통 range(1,4) 라고 한다면 1,2,3,4 를 뱉는 것이 아니라 1,2,3 즉, 마지막 숫자 전까지 return 한다.
따라서 range(1,4)는 갯수가 3개가 될 것이므로 take를 통해 lazy-sequence에서 3개를 end와 start를 뺀 만큼 가져온다.

3. iterate를 통해 lazy sequence를 1에서부터 하나씩만 (inc) 증가하도록 만든다.

자, 일단 자신이 lazy sequence를 모른다고 한다면 거수를 한번 해보고 거수한 인원들은 lazy 없이 설명을 봐도 좋다. lazy sequence는 함수형 프로그래밍에서 자랑거리로 들먹이고 다니는 것으로서 보통 '평가하기 전에는 사용하지 않는 sequence'라고 지칭하여 메모리를 덜 잡아먹는다는 장점을 갖고 있다고 한다.

마침 얘기가 나왔으니 functional programming의 자랑요소들에 대한 정리도 한 번 다루도록 하겠다.	
하지만 먼저 lazy evaluation이나 sequence에 대해서 궁금한 사람들은 아래 첨부한 유튜브를 보는 것을 추천한다.	
어떤 백인 소년이 자유롭게 그냥 카메라에다가 대고 자기 썰을 푸는데 range를 통한 설명이 은근히 기똥차게 느껴진다.

[![caucasian-boy](https://img.youtube.com/vi/QVU7IBGsJBo/0.jpg)](https://www.youtube.com/watch?v=QVU7IBGsJBo&t "Audi R8")

자, 여기까지가 나의 코드이고 과연 다른 행님들의 답은 어떻게 됐을지 봐보자.

### 1067's solution
```clojure
(fn [from to] ((fn iter [from to res] 
	(if (= from to) 
    	res 
        (iter from (dec to) (conj res (dec to))))) from to ()))
```
### 코드 설명
1. from과 to를 받는 fn은 또다시 from to res를 받는 함수를 받는다.
여기서 내부에 있는 from은 결국 같지만 파라미터 입장으로서 같은 것은 아니다.
맨 뒤의 input에 처음 fn의 from 과 to가 삽입되어 있는 것을 보면 알 수 있다.

2. 만약 from과 to가 같다면, res를 그대로 return하고
3. false라면 다시 iter라는 함수를 돌리는데 이때의 새로운 input값은 	
from => from 그대로 	
to => dec를 통해 1을 낮추고
res => 낮아진 to와 res를 conj를 하여 합친다.	

**즉 본 함수는 뒤에서부터 값을 삽입하는 구성을 띄며 맨 끝자리를 하나씩 낮춰가면서 from까지 담을 수 있는 함수라고 할 수 있다**

### maximental's solution
```clojure
#(take (- %2 %) (iterate inc %))
```
### 코드 설명
사실 이 함수는 내가 짠 함수랑 syntax만 다르지, 골자는 완전 같다.
1. take를 통해 lazy sequence에서 %와 %2를 뺀 만큼 가져온다.	
여기서 %와 %2는 첫 번째, 두 번째 파라미터를 뜻한다.
2. 가져오는 대상은 첫번째 파라미터에 inc iteration이다.

## 이상 전달 끝!
![dike-crying]({{ "/assets/img/pexels/dike_crying.jpg" | relative_url}})	
세상사 너무 복잡한 일들이 많다



