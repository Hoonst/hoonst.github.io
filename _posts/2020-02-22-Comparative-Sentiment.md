---
layout: post
title:  "Comparative Study of Deep Learning-Based Sentiment Classification"
categories: [paper]
---
# 딥러닝 기반 감성분석 비교 연구

감성분석에 대한 현황을 정리하고 싶어 

## Abstract

감성분석(분류)의 목적은 특정 문서가 긍정 또는 부정의 뉘앙스를 갖는지 파악하는 것이다. 감성분석은 다양한 산업 영역에서 많이 사용되며 상품에 대한 고객들의 의견 이해를 통해 상품과 서비스를 발전시키기 위해서 사용된다. 딥러닝은 다양하고 도전적인 영역에서 최신의 성과를 보이고 있다. 딥러닝의 성공가도로, 다양한 연구는 딥러닝 기반 감성분석 모델을 제의했으며 전통적인 머신러닝 모델보다 더 나은 성과를 보였다. 하지만 딥러닝 기반 감성분석의 실용적 관점의 이슈로 우수한 모델의 구조는 딥러닝에 사용되는 데이터의 특징에 영향을 많이 받는 것이다. 게다가 이마저도, 가능한 후보들 중에서 도메인 전문가의 지식에 의해 선정되고나, Grid Search로 결정된다.

> 모델에 영향을 주는 요인으로,
1. 훈련 데이터 특징
2. 도메인 지식
3. Grid Search (더 좋은 파라미터가 혹여나 있지 않았을까 하는 아쉬움)

이 논문에서는, 감성분류 모델을 구성하기 위한 유의미한 의의를 딥러닝 기반 감성분석 모델에 대한 비교 연구를 통해 나타내고자 한다. 특히, 여덟개의 딥러닝 모델(3 CNN / 5 RNN)과 두 개의 Input 구조(Word / Character Level)를 13개의 리뷰 데이터셋들을 통해 사용하고 다른 관점에 따라 분류 성능을 평가해겠다.

## Introduction

온라인 쇼핑의 성장에 따라, 신규 사업과 전통적인 기존 사업 간의 경쟁이 타오르고 있으며, 예로 온라인 매장을 개장하고 있는 백화점이나 슈퍼가 늘어나고 있다. 이런 e-commerce 산업에서, 많은 사람들이 구매한 물품이나 서비스들에 대하여 인기 있는 리뷰 사이트나 자신의 블로그, SNS에 표현하고 있다. 따라서, 소비자들은 구매 결정에 있어 다른 사람들의 온라인 평가로부터 큰 영향을 받고 있다.

결과적으로 온라인 리뷰 분석은 다른 경쟁자들에 대하여 비교 우위를 얻기 위하여 적절한 관리 전략을 결정하는 데 큰 도움이 되어, 판매 산업에서 리뷰 텍스트에 대한 연구는 큰 연구 분야가 되었다. 소비자들의 리뷰 데이터가 나날이 증가함에 따라 수작업으로 분석은 불가능하여 머신러닝 알고리즘이 대량의 리뷰 데이터를 분석하는 데에 사용되었다. 머신러닝의 감성 분석은 일반적으로 리뷰의 감성 분류를 뜻하지만, 의견의 양적인 추출, 그리고 희화, 이모티콘, 가짜 뉴스 판별과 같은 텍스트의 주관성을 함의하기도 한다.

감성 분석은 어휘기반, 머신러닝기반, 딥러닝 기반 모델로 나뉘어질 수 있다. 어휘기반 모델은 문서에 포함된 '감성 어휘'의 갯수로 감성 값을 나타낸다. 만약 긍정적인 감성 어휘가 부정적인 것보다 하나의 문서에 많이 존재한다면, 해당 문서는 긍정으로 분류되는 것이다. 이런 이유로, 감성어휘 사전이 감성분석에 사전에 준비되어야 한다.

여러개의 사전 연구들이 _SentiWords_, _MPQA lexicon_,_SentiWordNet_ 과 같이 이미 구축된 감성 어휘를 사용하였다. 머신러닝 기반의 모델은 라벨링이 된 문서들에 대하여 모델을 훈련시켰다. 라벨은 긍부정 또는 기존 값, 즉 Rating으로 지정될 수 있다. 모델의 성능은 사용된 알고리즘, 라벨 정확도, 그리고 라벨 문서의 갯수에 따라 다르다. 감성 분석에 사용되는 머신러닝 알고리즘은 Naive Bayes, SVM이 있다.

최근에는 CNN or RNN과 같은 딥러닝 기반 감성분석이 많이 활용되며 여러 연구에서 어휘기반, 머신러닝기반의 성능을 한참 뛰어 넘고 있다.

이렇게 좋은 성과를 보이는 딥러닝기반 감성분석이지만, 각기 다른 도메인과 데이터셋에 대한 최적의 구조는 존재하지 않아, 적용을 하려는 사람들이 자신들의 데이터에 대하여 어떤 딥러닝 구조를 활용해야 할지에 대한 판단이 어렵다.

이에, 우리는 다양한 딥러닝 모델을 비교하는, 구조적으로 디자인된 연구를 통해 모델의 구조/ 데이터셋 특징과 분류 성능에 대한 관계를 보이고자 한다. 결과적으로, 3개의 CNN, 5개의 RNN 기반 모델을 사용해 8개의 구조를 고려하였다. 각 모델에 대하여 또한, word-level과 character-level input이 비교되어 13개의 다른 감성 분석 데이터 셋에 대하여 연구되었다.

우리는 본 연구에 대하여 3가지 핵심적인 질문에 대한 답을 얻고자 했다.
1. 리뷰 데이터셋에 따라 감성 분석이 어떻게 달라지는가?
2. CNNs and RNNs과 같은 모델들의 `기본적인` 구조는 어떻게 감성분석에 영향을 주는가?
3. Word-level과 Character-level은 감성 분석 성능에 어떤 영향을 주는가?

앞에서 언급한 sarcasm이나 fake-news detection 같은 것이 작가의 감성을 이해하기 위하여 중요하지만, 공개 데이터로는 복잡한 딥러닝을 구성하기에는 무리였다. 그 이유는 극값 분류가 감성분석에서 가장 근간적이고 널리 진행하는 목적이었기에 해당 목적에 대한 집중이 필요했다.

논문의 나머지는 다음과 같이 구성된다.

**Section 2**: 전통적 기법과 딥러닝기반 감성 분석 연구들에 대하여 간단히 리뷰

**Section 3**: 선택된 딥러닝 모델, 데이터셋, 실험 디자인

**Section 4**: 실험 결과

**Section 5**: 향후 연구

## II. 연관 연구
### A. EARLY HISTORY OF SENTIMENT ANALYSIS

초기의 감성분석 연구는 인간의 지능을 양적으로 연구하기 위한 인지 심리학 연구에서 활용되었다. 하지만 최근의 많은 연구들은 인터넷의 상품이나 서비스에 대한 Ratings와 그에 대한 리뷰 텍스트에 대한 쉬운 접근을 통해 형성된 대량의 Labelded dataset에 대하여 통계적, 그리고 머신러닝 모델을 갖추고 있다. 예를 들어, Nasukawa and Yi는 Syntactic parsing(언어 구조적 파싱)과 감성어휘 추출을 통하여 온라인 웹페이지의 감성 분류를 진행했다. Yu and Hatzivassiloglou는 온라인 뉴스에 대하여 문서의 감성에 대한 예측 뿐만 아니라, 각 문장에 대한 감성 값도 추출해내었다. 일련의 연구들은 변수 조정과 다양한 머신러닝 모델을 통하여 온라인 리뷰에 대한 모델 성능 향상에 힘썼다.

### B. DEEP LEARNING FOR SENTIMENT ANALYSIS

### CNN
딥 뉴럴 네트워크 기반 분류 모델이 여러 도메인(CV, NLP)에서 기존 모델보다 더 나은 성능을 보이자, 감성 분석에도 사용되기 시작하였다. CNN과 RNN은 아주 전형적으로 사용되는 기본적인 구조이다. CNN은 FeedForward Matrix 안의 Vector 대신, Matrix 또는 Tensor를 입력값으로 가정한다. Convolution의 작동 시, receptive field라고 알려진 submatrix가 사용되어 receptive field와 convolution filter 사이의 element-wise 곱을의 합으로 scalar 값을 산출해낸다. 이 Convolution 과정은 input matrix/tensor의 좌상부터 우하까지 filter를 넘나들면서 진행된다(stride). Convolution의 크기는 filter의 갯수, 그리고 convolution의 stride, 그리고 hyperparameter로 결정되낟. Pooling은 convolution layer의 결과값 size를 줄이는 데에 사용되며, 특정 구역의 평균 또는 최대값으로 계산한다. CNN은 본디 컴퓨터 비전쪽으로 개발되었으나, 최근에는 text 분석에 많이 활용된다. 일련 최근의 연구에서 CNN이 언어의 위계적 구조를 학습할 수 있음을 보였고, 변수의 길이를 효과적으로 다룰 수 있음을 시사하였다. Kalchbrenner et al.은 두 Convolution과 pooling layer들로 Stanford Sentiment Treebank(SST)와 Twitter sentiment datasets에 대한 실험을 진행하였다. 실험 결과는 CNN이 SVM이나 feedforward 신경망보다 더 나은 결과를 보였다고 한다. 다른 연구들에서는 더 많은 convolution과 pooling layer들을 추가하고 drop-out과 같은 정규화 기술을 통해 성능을 높였다.

CNN의 성능이 층의 갯수가 증가할 수록 증대된다고 소개가 되었으며, 이는 그렇게 디자인된 아키텍처가 loss function에서 부터 input matrix/tensor까지의 Gradient flow를 촉진시키기 때문이라 했다. 하지만 Kim은 감성분석에서 단 하나의 convolution과 수 백개의 filter들을 포함한 얕고 넓은 CNN 구조 높은 성능을 나타냄을 보였다. Le et al.은 추가적인 실험으로 shallow-and-wide 구조가 깊은 구조보다 더 나은 성과를 창출함을 보였다.

지금껏 소개된 연구들은 word를 input matrix를 구성하기 위한 기저 unit으로 사용했다. Input matrix의 넓이는 word vector size에 연관되었고, 높이는 문서 속의 높이와 연관되었다. Representation에 있어 ith 행은 문서의 ith 단어에 해당하였다. Zhang et al.은 최초로 word-level input 대신에 character-level을 사용해 CNN을 구성하였다. Kim은 단순히 하나의 convolution과 하나의 FC-layer를 활용했으나, Zhang et al.은 6개의 Convolution layer를 통하여 text feature를 추출하고, 3개의 FC-layer를 통해 문서를 분류하였다. 그들은 AG corpus, Sogou News, DBPedia, Yelp Review, Yahoo! Answers dataset 그리고 Amazon review를 통해 제안된 모델에 대한 성능을 평가하였다. 결과는 Yelp Review와 Amazon review에 대하여 Binary 분류 과제에서 평균 5%의 에러률을 나타냈다. Conneau et al.은 CNN을 이전보다 훨씬 더 깊게 제작하여 활용했다. 이때 29층의 Simonyan and Zisserman에 의해 제안된 CNN의 구조를 활용하였다. Zhang et al.과 같은 데이터셋을 활용한 실험에서 binary 분류 과제에서 평균 4.5%의 에러률을 보였다고 한다.

### RNN

RNN은 연속적인 데이터를 처리하기 위한 딥러닝 모델이다. 기본적으로 텍스트는 연속적인 데이터이기 때문에, RNN은 텍스트 분석에 많이 활용된다. 하지만 Sequence를를 처리하기 위한 RNN의 recurrent structure는 long-term dependency에 의하여 gradient가 사라지거나 폭발하는 문제를 야기하곤 한다.

**Hochreiter and Schmidhuber and Cho et al.:**

LSTM cell 또는 GRU라고 불리우는 Gate unit를 삽입함으로써 위의 문제를 해결
LSTM이나 GRU는 '기억'의 역할을 하는 cell이 없는 vanilla RNN보다 높은 성능을 보인다고 한다. 하지만 그들의 성능은 크게 차이나지 않았다. GRU는 LSTM의 특별한 케이스로서 LSTM의 forget gates와 input을 합침으로써 파라미터의 수를 줄이는 단순화한 LSTM이다. 많은 연구들은 LSTM 또는 GRU를 감성분석에 활용하고 지속적으로 좋은 성과를 나타낸다.

### C. COMPARATIVE STUDIES ON SENTIMENT ANALYSIS BASED ON DEEP Learning

많은 연구들이 CNN 또는 RNN을 기반으로한 감성분석 DNN을 제안했지만, 단지 적은 연구들만이 다양한 딥러닝 기반 감성 분류 모델에 대한 성능을 구조적으로 비교하였다.

Hu et al.:  
딥러닝 기반 모델이 사전, 어휘기반 / SVM / naive Bayes보다 감성분석에서 더 높은 성능을 보임을 나타냈다. 하지만 그들은 F1 score나 정확도와 같은 '양적인' 성능 지표를 제공하지 않았다.

Yin et al.:  
CNN, LSTM, GRU의 감성 분류 성능을 비교하였다. 하지만 그들은 모델 구조의 다양성에 대한 고려를 하지 않았고, 단 하나의 데이터셋에 대한 실험이었기 다소 제한적인 결과를 나타낸다.

Ouyang et al. and Singhal and Bhattacharyya:  
Basic CNN과 RNN 구조를 비교. 하지만 input type과 모델 구조에 대한 다양성을 포함하지 않았다.

Katic and Milicevic:  
CNN과 LSTM 성능을 Amazon review dataset에 대하여 평가하였다. 하지만 실험 setup을 충분히 기재하지 않았기 때문에 독자들이 실용적인 가이드라인을 얻지 못했다.

Zhang et al.:  
CNN, RNN을 포함하는 딥러닝 구조를 통해 9개의 연구를 제시.
하지만 성능 비교는 제공되지 않았다.

따라서 본 연구에서는 구조적으로 디자인된 비교 연구를 8개의 아키텍처와 2개의 input type을 13개의 review dataset을 통해 진행해보고자 한다.

## III. MODELS

이번 섹션에서는, 간단히 8개의 벤치마킹한 모델들을 설명한다:  
CNN 모델 3개, RNN 모델 5개

### A. CONVOLUTIONAL NEURAL NETWORK (CNN)-BASED MODELS

1) ONE-LAYER CNNs  
선정된 첫 모델은 Figure 1a에서 볼 수 있는, 하나의 convolution layer를 갖는 CNN model이다.

![Figure1](../img/sentiment_analysis/image_1.png)
**Figure 1. Architecture of convolutional neural network (CNN)-based models**

이때 이미지 처리에서 활용되는 `2차원`의 convolution filter대신 `1차원`의 것을 사용한다. Convolution을 활용하여 local feature를 추출하고자 할 때, 수직과 수평의 공간 정보가 중요하다. 결과적으로 상하좌우 방향으로 receptive field를 가로지르며 사각의 convolution 절차를 거쳐야 한다. 하지만 input matrix의 각 행이 단어나 문자의 distributed representation이기 때문에 단순히 수평적 공간 관계가 정보를 담고 있다. 따라서, input matrix와 넓이가 같은 직사각형 구조의 convolution filter가 사용된다. Convolution filter와 input matrix의 넓이가 동일하기에, vertical striding만이 필요하다. 이 구조에서는 input sentence n은 고정 변수이고, 아래와 같이 표현된다.

`+` 기호는 concatenate(덧셈이 아닌 붙이는 과정) 연산이며, xi는 문장의 ith 단어이다. 만일 문장이 고정된 길이보다 짧다면, 제로 패드가 input matrix의 끝에 추가된다. Feature c가 연속적인 h개의 단어와 필터를 활용한 convolutional operation에 의해 생성된다. 예를 들어 c1은 아래와 같이 생성된다. (b는 편향)

w는 convolution filter의 가중치이며, f는 비선형 함수를 뜻한다. ci 집합은 c = [c1, c2, c3 ... ci]를 구성한다. 각 Feature map에 대하여, max pooling이 c에 적용되어 Feature map의 최대값을 얻어내며, 이는 각 convolution filter의 가장 중요한 단어를 추출함을 위해서이다. 특히 100개의 feature map이 3,4,5 / 3개의 filter size로 사용되었다.

Max pooling이 끝나게 되면, 300 차원의 벡터가 생성이 되며 두 개의 노드(긍/부정) output layer로 fully connected되었다. 마지막 hidden layer와 output layer 사이에 droup out이 적용되어 모델의 복잡도를 정규화하였다. Input matrix를 고려하여, Kim은 단어 벡터에 대하여 4개의 전략을 사용하였다.

CNN-rand model: 단어 벡터는 랜덤으로 initialized 되었으며 다른 신경망 파라미터와 함께 훈련되었다.

CNN-non-static: Initialization 후에 word2vec 기법으로 fine-tune

CNN-static: non-static과 다르게 word-vector가 변하지 않는다

실험 결과, 세 모델의 결과 차이가 크지 않아 CNN-rand 모델을 비교의 대상으로 사용하였다.

2) NINE-LAYER CNN  
Zhang et al.은 6개의 convolutional layer들과 뒤이어 3개의 fully connected layers로 감성 분석을 진행하였고, Kim의 것보다 훨씬 깊었다(Figure 1b). 이 아키텍처의 핵심은 temporal convolution이다.

이산 input function 인 g(x)와 이산 kernel function인 f(x)가 존재할 때, h(y)는 다음과 같이 정의한다.

### B. RNN-BASED MODELS
RNN이
