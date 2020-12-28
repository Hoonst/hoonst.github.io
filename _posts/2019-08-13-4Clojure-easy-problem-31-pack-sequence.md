---
layout: post
title: "(4Clojure) Easy_Problem_31_pack_sequence"
categories: [clojure]
comments: true

---
# Pack a Sequence
 
Difficulty:	Easy
Topics:	seqs

Write a function which packs consecutive duplicates into sub-lists.
<!--more-->
```clojure
(= (__ [1 1 2 1 1 1 3 3]) '((1 1) (2) (1 1 1) (3 3)))

(= (__ [:a :a :b :b :c]) '((:a :a) (:b :b) (:c)))

(= (__ [[1 2] [1 2] [3 4]]) '(([1 2] [1 2]) ([3 4])))
```
이번 문제는 당황스럽게도 쉽게 풀렸다. 호오옥시? 하면서 작성한 함수가 바로 적용이 되었기 때문이다.따라서 정답이라는 메시지와 함께 바로 넘어갈 수 있었다. 넘어 갔어야 했다...

먼저 이전에 비슷한 경험이 있어서 기억난 function으로 푼 나의 풀이를 살펴보자

```clojure
partition-by identity
```
끝이다. 너무 단순해서 진짜 easy 난이도에 easy문제도 있구나 라고 생각하며 고수 행님들도 비슷하게 문제를 풀어나갔을 거라 생각했다. 하지만 뭔가 많은 것을 배울 수 있도록 해주려고 하는 듯 싶었다. 

![dike-thinking]({{ "/assets/img/pexels/dike_thinking.jpg" | relative_url}})	

자 그럼, 1067 형님의 답부터 살펴보자	

** 상당히 코드가 빡셉니다 / 부가 설명이 많습니다 **

### 1067's solution
```clojure
#(->> %
     (reduce (fn [[agg prev-l] n]
               (if (= (first prev-l) n)
                 [agg (cons n prev-l)]
                 [(conj agg prev-l) (list n)])) [[] ()])
     (apply conj)
     (drop 1))
```
### 코드 설명
맨 처음부터 절망이다. ->>가 나왔기 때문이다. 이것은 threading  macro라고 하는데 threading은 커녕 macro도 아직 개념이 익숙하지 않다. 하지만 알 수 있는 것은 특정한 기능을 한다는 건데 threading macro에는 ->> 이외에 ->도 존재한다. 

간단하게 말하자면 -> 와 ->> 모두 인자를 함수 묶음에 전달하여 순차적으로 실행한다는 공통점이 있다. 그런데 왜 굳이 이 두 화살표의 생김새를 다르게 해놓았을꼬? 

정말 놀랍게도 이것을 한국어는 커녕 명확하게 써준 포스트가 없었다 (내 기준). 하지만 많은 글들 및 stack overflow를 많이 살펴본 결과 가장 마음에 드는 답변은 이거였다. 

![threading-macro]({{ "/assets/img/pexels/stack_overflow_threading_macro.jpg" | relative_url}})	

clojure에 아직 경험이 적거나, 익숙하지 않은 사람들은 와닿지 않을 수 있다. 나도 아직 온전하게 백프로 이해한 것은 아니지만 정리 & 번역을 시작해보겠다.

### ** -> Thread first macro**
함수가 단일 대상에 대하여 적용이 되는 함수라면, 그 단일 대상은 first argument이다. 예를 들어 sequence에 '단일 값'을 합치는 conj 함수나 대체와 비슷한 역할을 하는 assoc 같은 경우 역시 '단일 값'을 대상으로 한다. 따라서 **->** 를 사용하는 경우는 뒤에 따라오는 함수들이 모두 first argument를 사용하는 함수여야 한다.	

### ** -> Thread last macro**
반면 함수가 단일 대상이 아니라 sequence에 대하여 적용이 되는 함수라면 그 대상은 last argument이다. map과 filter 같은 경우는 하나의 대상이 아니라 sequence에 있는 모든 대상에 대하여 적용하지 않는가? 따라서 sequence에 대한 연속적인 함수 사용은 **->>** 를 사용하게 된다.

또 이 둘의 개념이 잘 이해가 되지 않는다면 단순하게 구분할 수 있는 방법은 ->/ ->> 뒤에 오는 인자들의 위치이다. **->**  같은 경우 다음 함수들에서 함수 바로 앞에 인자들이 위치하며 **->>** 같은 경우는 맨 뒤에 위치하게 된다. 위 사진의 예시에서 
* -> 의 (conj 4) == (conj [1 2 3] 4)
* ->> 의 (map inc) == (map inc [1 2 3])

macro에 대한 설명이 끝났지만 우리는 이제야 1번째 줄을 끝낸 것이다.	
![banana-shock]({{ "/assets/img/pexels/banana_shock.jpg" | relative_url}})

그럼 다음 줄을 살펴봐보자. Reduce에 대한 심화 학습 시간이다.
```clojure
(reduce (fn [[agg prev-l] n]
               (if (= (first prev-l) n)
                 [agg (cons n prev-l)]
                 [(conj agg prev-l) (list n)])) [[] ()])
```
이 reduce function을 보고 한번에 이해했으면 똑똑이가 진배 없고 ->>에 대한 이해를 깊게 하고 있는 편이다. 또한
> 아니 왜 인자는 [[agg prev-l] n] 인데 주어지는건 왜 [[] ()] 로서 agg / prev-l 밖에 없는거야?	

라는 의문을 품을 수 있다. 합리적인 의심이다. 하지만 방금 threading macro를 배웠다는 것을 까먹지 말길 바란다.
현재 우리는 **thread last macro**를 사용하고 있으므로 전달 받은 인자는 reduce 함수의 맨 끝에 들어가게 된다.
따라서 reduce를 사용하는 익명함수에 들어가는 n은 위의 %에서 들어가게 된다. %가 sequence일테니 sequence 인자 하나하나씩을 함수에 사용하게 된다.

### 코드 설명
1. reduce를 사용하는 익명함수는 총 두개의 input을 받게 되고 하나는 initial value인 [agg prev-l] 하나는 input value인 n이다.
2. 만약 prev-l의 first가 n과 같으면 initial value [agg prev-l]은 각기 [agg(그대로) (cons n prev-l) n과 prev-l로 새로운 seq를 만든다.]
3. 아니라면 initial value는 [(agg와 prev-l을 conj) (prev-l은 n으로 이루어진 list)]

상당히 복잡할 거라고 굳게 믿는다! 그래서 도움이 될 지도 모르기 때문에 pseudo code마냥 [1 1 2 1 1 1 3 3]을 input으로 한 예시를 적어보았다.

![doodle]({{ "/assets/img/pexels/doodle_on_reduce.jpeg" | relative_url}})

여기까지 왔다면 그 다음은 꽤 쉬울 것이다. 위에서 재난 두 가지를 마주했기에 속이 좀 편할 것입니다요...
```clojure
(apply conj)
     (drop 1)
```
### 코드 설명
이전에 apply에 대해서도 할 말이 있다. python이나 r을 사용해보았다면 clojure의 map이 python에서의 apply와 비슷하다는 느낌을 받았을 수 있다. 하지만 clojure에서의 apply는 map과 다르다 (다르니까 이름이 다르겠지 싶다). 또한 apply는 reduce와도 상당히 헷갈리는데 아래의 예시를 봐보자.
```clojure
(apply + [1 2 3 4 5])

(reduce + [1 2 3 4 5])
```
둘의 결과는 모두 15이다. 그래서 혹자는 둘이 같은 기능을 하는 것이 아닌가? 라는 오해를 할 수 있는데, apply는 +를 [1 2 3 4 5] 에게 적용 (1+2+3+4+5)와 같은 느낌을 주고, reduce는 +를 ((((1+2)+3)+4)+5) 이런식으로 적용한다.
map과 apply의 차이는 다음의 stack overflow를 살펴봐보자 

![map-apply]({{ "/assets/img/pexels/map_apply_stack_overflow.jpg" | relative_url}})

1. apply를 사용해 return값으로 다온 두 seq를 합치고
2. 처음 것을 삭제한다 (실제로 모든 seq에 대해서 이 함수를 돌리면 첫번째 값은 ()의 빈 seq라서 지우는 것이다)

수고하셨습니다만... 아직 maximental 형님이 남았습니다...

### maximental's solution
```clojure
(fn p 
  ([[h & t]] (p [] [h] t))
  ([a h t] 
    (if-let [[f & n] t]
      (if (= (first h) f)
        (p a (conj h f) n)
        (p (conj a h) [f] n))
      (conj a h))))
```

### 코드 설명
이번 maximental 형님의 코드는 간단한 syntax들이 많다. 하지만 실상 들여다보면 의문 투성이라 고통스럽다.
(if-let)만 알면 다른 구문들은 코드 설명에서 flow대로 설명할 수 있을 것 같다.

(if-let)은 말 그대로 if와 let을 합친 함수이다. 즉 let에서 입력을 받는 피동의 대상인 왼쪽이 아닌 오른쪽에 true일 경우 let의 기능을 하는 것이다. 이는 nil이나 false를 걸러내기 위한 함수이다.

1. fn p는 단순히 익명함수의 이름이 p라는 뜻이다. 처음에 이것도 input인 줄 알고 당황했다.
2. [[h & t]] 같은 경우 input으로 vector가 들어오고 first를 h, rest를 t로 나눈 것이다.
3. 이런 h와 t를 p function에 다시 집어 넣고 추가적으로 []를 넣는다. 이는 각기  a / h / t로 나뉜다.
4. 만약에 t가 nil이 아니여서 if-let에 true가 된다면 t 역시 분해하여 [f & n]으로 만든다.
5. 만약 h의 first가 f와 같다면 다시 p를 실행하는데 이때 input은 (a (h와 f의 혼합) n 이다)

여기까지 설명하면 뒤에는 어떻게 돌아가는 지 감이 올 수 있다. 따라서 나는 여기까지 적도록 하고 한번 직접 위에서 내가 pseudo-code식으로 적었던 거처럼 직접 적어보길 바란다.

### 이상 전달 끝!!
후 힘들었다.... 머리가 아파옵니다
![lot-hurt]({{ "/assets/img/pexels/lot_hurt.jpg" | relative_url}})