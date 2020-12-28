---
layout: post
title: Clojure, 뭐가 그래 좋은가?
categories: [clojure]
comments: true

---
# Clojure - 뭐가 그리 좋은가? 나는 잘 모르게따.

### 클로저(Clojure)란?

** 해당 포스트는 Brian Will Youtube Channel의 Clojure 강의를 번역한 것입니다 (하단 링크 첨부). **

프로그래밍 언어 클로저(Clojure)는 2007년, Rich Hickey에 의해 만들어졌다. 클로저의 가장 첫 번째로 오는 특징은 바로 Lisp의 분화라는 것이다. Lisp는 1950년대에 John McCarthy에 의해 만들어졌으며,<!--more--> 현대 프로그래밍에서 자주 언급이 되는 일급 객체와 함수, 가비지 컬렉션 등의 개념의 근간의 역할을 하고 있다. Lisp 그 자체는 초기에는 많이 그리고 오래 사용되지는 않았지만 많은 모방꾼(imitators)들이 존재했다. 이런 모방꾼들을 macro를 Lisp 내로 들여왔는데 macro란 source code를 input으로 받고 output을 source code로 내뱉는 것이라고 한다 (macro는 나중에 다룬다). Non-Lisp 언어들도 macro의 역할을 하는 것들이 종종 존재하지만 그것이 Lisp에 존재하는 macro와 비견할 수 없다고 한다. C언어에도 'Preprocessor'가 존재하지만 실상 사용하기에 syntax가 너무 사용하기 힘들고 복잡해서 잘 사용하지 않는다 한다. 하지만 Lisp의 Syntax는 매우 쉬워서 이 macro 작성이 수월하다.

### 하지만 비단 위와 같은 이유로 Clojure가 개쩐다고 말하기는 부족하다.
클로저의 우수성은 Functional programming(이하 FP)에 대한 접근법에 있다. 많이들 헷갈려하고, 특히 내 자신이 아직 잘 개념이 확립이 안되었기 때문에 이 글을 통해서 한 번 다시 정리하고자 한다. 

### 함수형 프로그래밍이란?
함수형 프로그래밍은 우리가 흔히들 알고 있는 명령형(Imperative) 프로그래밍과 대조를 이루는 프로그래밍 형태이다. 이 둘의 핵심적인 차이는 바로 변할 수 있는 변수, 즉 mutable state를 어떻게 다루느냐이다. 만약에 변수에 새로운 값이 대입될 수 있다면, object field가 변한다면, array의 값들이 변한다면 mutable state라고 부를 수 있다. 명령형 프로그래밍에서는 언제든지 우리가 변수에 원하는 값을 대입하여 변경할 수 있지만 함수형 프로그래밍에서는 최대한 이 변하는 상황을 모면하고자 한다. 그 이유는 함수에서 mutable state를 사용하지 않는다면 참조투명성(referentially transparent)을 갖기 때문이다. 참조 투명성이라는 뜻은 같은 input에는 항상 같은 output을 내놓는다는 뜻이다. 이는 다른 말로 pure function이라고도 부른다. Impure function은 같은 set의 argument가 제시된다고 해도 다른 return value를 내뱉을 수 있는데 function이 외부의 mutable state (ex. global variable) 에 의존할 수 있기 때문이다. 또한 역으로 function이 외부의 mutable state를 변화시킬 수도 있다.

이것이 함수형 프로그래밍의 골자이며 mutable state가 존재하지 않음으로써 비로소 함수가 truly modular(독립적인?) 하게 되는데, 왜냐면 하나의 함수를 관리함에 있어 다른 함수의 눈치를 안보고 독자적으로 활동할 수 있기 때문이다. 

그런데 이런 의문이 들 수가 있다.
> 아니, 물론 부수효과가 존재한다고 해도 Mutable state가 편하지 않나? 

이것은 우리가 FP로 새로운 변수를 만들 수 있음을 지각하지 못할 때 나올 수 있는 생각이다. 함수형 프로그래밍에서의 변수에 새로운 값을 할당하는 일은 '기존 변수'에 몇가지 변형을 가한 '새로운 Copy'를 만들어내는 것이다. 즉 기존 변수를 조작하는 대신에 (기존 변수의 값 + modification)의 새로운 변수를 만들어낸다고 보면 된다.

예를 들어,
```clojure
[1 2 3] -> [1 7 4] (a new, seperate list)
```
기존의 [1 2 3]을 [1 7 4]로 바꾸려면 어떻게 하면 되겠는가? 기존의 imperative programming이라면 두 번째 숫자를 2에서 7로 바꾸면 된다.
하지만 FP에선 2를 7로 바꾼 전혀 새로운 list를 만들게 되어 mutable state가 발생할 여지를 없애버리게 된다. 

> 하지만 당연히 가져야만 하는 의문은... 그럼 비용이 너무 과하지 않나용?? 

> 만약 [1 2 3]을 [1 2 4]/[1 2 5]/[1 2 6] 이런식으로 변형을 여러 번 준다고 하면 중복 데이터가 너무 많아지게 되지 않는가?

이것을 해결하기 위하여 Persistent Collection(이하 PC)이 등장하게 된다. PC는 immutable을 지원하나 Copy를 쉽게 만들어주는 역할을 하게 된다. 간단하게 말하자면, PC는 기존의 변수와 새로운 변수가 공유하고 있는 값들은 Memory에서 Share하게 된다. 예를 들어

```clojure
[1 2 3 4 5 6 7 8 9]
--> [1 2 3 4 5 6 7 8 9 "HI"]
```
위와 같이 기존 list를 새로운 list로 바꾸게 될 때 바뀌지 않은 부분의 데이터는 공유하게 되고 새로운 것만 갖는 변수의 Copy를 갖기 때문에 Memory에 부하를 주지 않게 된다. Persistent Data structure에 대해서 좀 더 자세히 이해하고 싶은 사람들은 다음 link를 읽어본다면 꽤 괜찮은 답을 얻을 수 있을 것이다. https://hackernoon.com/how-immutable-data-structures-e-g-immutable-js-are-optimized-using-structural-sharing-e4424a866d56

### 하지만 이것은 너무 이상적인 예시였다.
사실 현실에서는 mutable을 극복하고 피하는 것이 쉽지많은 않다. I/O의 세계를 살펴보기만 해도 (Files, network resources, user interactions, etc) 수많은 mutable state들이 존재한다. 파일을 읽고 그것을 화면이 띄우는 활동도 모두 mutable state를 다루는 것이다. Haskell이나 Scala 같은 프로그램과 같은 함수형 프로그래밍 언어에서는 악명높은 'monad' 데이터 타입을 사용해서 이 문제를 극복하고자 한다. 

하지만 Clojure에는 이에 대한 해결책이 딱히 없다. 그리고 있어야 할 필요가 크게 없는 것이 함수형 프로그래밍은 mutable state를 제한하는 것이지 없애려고 혈안이 되어 있는 프로그래밍이 아니다. 필요할 때는 그것을 사용할 때가 존재하기는 한다. Clojure는 순전히 프로그래머에게 pure code의 비율이 impure보다 많을 것을 추천할 뿐이다. 따라서 Clojure에서도 일정 량의 mutable data가 존재하기는 하지만, 그의 존재는 multiple thread에서 발생하는 실행이 매우 까다로워지는 것이다. 각각의 thread execution은 서로를 망칠 수 있는데 그 이유는 하나의 thread의 행동이 다른 thread의 행동이 예상하는 방향으로 진행되지 않을 수 있기 때문이다. 

이것을 해결하기 위해 Reference type을 내놓는데 이는 추후에 설명하도록 한다.

### Java의 연동성
Clojure는 기본적으로 Compile시 Java bytecode로 진행하며 JVM상에서 구동되고 Clojure data type이 Java class에 의해 정의된다. Clojure는 Java와의 연동성이 매우 좋아 Java library로 일하는 경우가 많다. 

### No (encapsulation, inheritance) but yes polymorphism
Clojure 개발자인 Rich Hickey는 OOP를 거의 경멸 수준으로 바라봤기 때문에 Java와 Clojure의 깊은 관계에서도 OOP 특성인 encapsulation과 inheritance를 데이터 타입에서 제외하였다. 하지만 그 중에서 polymorphism은 좋아했기 때문에 추가했다고 한다. 이로 인해서 함수를 정의할 때 argument arity가 달라짐에 따라 행동을 다르게 할 수 있게 만들 수 있는 것이다. 

> Brian Will Youtube Link
[![clojure-rocks](https://img.youtube.com/vi/9A9qsaZZefw/0.jpg)](https://www.youtube.com/watch?v=9A9qsaZZefw&list=PLAC43CFB134E85266 "cljclj")

https://www.youtube.com/watch?v=9A9qsaZZefw&list=PLAC43CFB134E85266


