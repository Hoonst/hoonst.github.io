---
layout: post
title: "Point-Generator"
categories: [paper]
comments: true
---

#Get To The Point: Summarization with Pointer-Generator Networks

#초록
* seq2seq의 적용으로 abstractive summarization이 활성화
* 2개의 큰 단점:
> 1. 디테일 부분에서 오류가 발생
> 2. 같은 단어를 반복하는 경우가 생김

> <b>1번 문제</b>:
Pointer-generator network를 통해 pointing을 하여 source text의 단어들을 copy 해온다 (Extractive)
> <b>2번 문제</b>:
Coverage를 통해 무엇이 요약에 사용되었는지 기록하여 반복 횟수를 낮춘다.

* CNN / Daily Mail 요약 과제에서 SOTA보다 2-ROUGE가 앞선다


## 2. Our Models
* (1) Baseline seq2seq model
* (2) Pointer Generator model
* (3) Coverage Mechanism

## 2.1 Seq2seq attentional model
모델 진행 순서:
* 1. Word Token $w_i$가 일대일로  single layer bidirectional LSTM에 태워,
encoder hidden state $h_i$를 만든다.
* 2. 각 t step에서 decoder(single layer unidirectional LSTM)은 이전 단어의 embedding과 decoder state $s_t$를 얻는다.
* 3. Attention distribution $a_t$는 Bahdanau et al.의 방식을 따른다.
     Attention distribution은 decoder가 다음 단어를 예측할 때 어느 쪽을 바라봐야 하는지 알려준다.
* 4. Encoder hidden state와 attention distribution과 가중합을 진행해 $h_t$ 생성.
* 5.

## 2.2 Pointer-generator Network
Pointer-generator network는 baseline(seq2seq)와 pointer network의 혼합이다. 
이는 pointing을 통한 단어 복사, 그리고 고정된 어휘 집합에서 단어를 생성한다.

다음 단어는 copy일지, generate일지 결정하는 확률은 $p_{gen}$으로 결정하며 해당 값은
* context vector $h_t$
* decoder state $s_t$
* decoder input $x_t$
로 생성된다.

<img src = '../img/point_generator/p_gen.png'>

각 문서에서 extended vocabulary는 source document의 모든 단어와 전체 단어 집합의 합집합이다.


# 2.3 Coverage Mechanism

Seq2seq에서 또 흔한 문제는 단어의 반복이다. 이는 이전 Decoder에서 사용한 attention 분포의 합으로 나타낸다.

이는 처음의 attention 식에 coverage vector를 추가하여 변형시킬 수 있게 된다.
