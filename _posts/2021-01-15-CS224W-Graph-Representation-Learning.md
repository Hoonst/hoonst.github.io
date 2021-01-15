---
layout: post
title: "CS224W Lecture 7"
description: "Graph Representation Learning"
tags: [CS224W]
date: 2021-01-15
comments: true
typora-root-url: ../../hoonst.github.io
---

CS224W 7번째 강의 정리를 정리해보았습니다. 강의에서 사용된 이미지 전체는 CS224W 수업 자료에서 가져왔음을 미리 밝히는 바입니다. 

강의 Link: [Graph Representation Learning](https://www.youtube.com/watch?v=4PTOhI8IWTo)

강의 Slide: [CS224W: Fall 2019](http://snap.stanford.edu/class/cs224w-2019/)

# Intruduction

본 챕터는 Representation Learning을 Graph에 적용하는 방법에 대해서 설명합니다. Representation Learning이란 데이터를 다른 차원 (보통 저차원)으로 Mapping하여 Represent하는 학습을 말하며, 가장 친숙한 예시로는 Word2Vec (단어 $\Rightarrow$ 벡터)입니다. 대부분 기법 이름 뒤에 '2vec'이라는 단어가 붙는다면 Representation Learning일 가능성이 매우 높습니다.

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210113213451164.png" alt="image-20210113213451164" style="zoom:67%;" />

위의 그림은 Machine Learning 알고리즘을 학습함에 있어서 진행되는 보편적인 절차입니다. Raw Data를 가지고 정형 데이터로 만들어 학습 알고리즘을 구성해 모델을 구성한다는 절차를 표현하고 있습니다. 어떤 프로젝트를 시작함에 있어서 가장 비용과 시간이 많이 드는 부분은 Raw Data를 정형 데이터로 변환하는 과정이며, 이 때 Feature Selection / Extraction과 같은 Feature Engineering을 진행합니다. 확실히 이는 좋은 방법이지만, 결국 더 편리한 전처리를 위하여 Representation을 대신 사용합니다. Real Data는 고차원인 경우가 많은데 이 중 어떤 Feature를 사용할지에 대하여 고민할 것이 아니라, 다수의 Feature들을 좀 더 작은 차원으로 줄여서 표현하는 방식을 취합니다. 따라서, 자동적으로 Feature를 압축하여 학습합니다. 

어떻게 보면 PCA와 같은 Feature Extraction처럼 압축한다는 의미가 담겨있지만, 불필요한 요소를 잘라버려서 차원을 축소하는 느낌과 완전히 한 데 모아두고 섞어버리는 것과는 다소 의미의 차이가 있습니다. 

# Feature Learning in Graphs

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210113215134692.png" alt="image-20210113215134692" style="zoom:67%;" />

그래프의 Feature를 학습한다는 것을 그림으로 표현하면 위와 같으며, 하나의 Node의 Feature를 저차원 벡터로 임베딩 하는 것입니다. 즉 고차원의 Feature를 저차원의 Feature로 Embedding / Mapping 하는 것입니다. 

하지만 이런 Embedding 기술을 Graph에 적용하기가 다소 까다로운 부분이 있습니다. Embedding을 하기 위한 Deep Learning 기술들은 단순한 Sequence(NLP)나 Grid(Image)의 데이터에 적용되도록 발전해왔기 때문에 Graph에 적용하기 위해선 그에 걸맞는 변형이 필요하기 때문입니다. 

# Embedding Nodes

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210113220132381.png" alt="image-20210113220132381" style="zoom:67%;" />

 노드를 Embedding한다는 뜻은 무엇일까요? 이는 위의 그림과 같이 Network가 존재하고 노드별로 Embedding Space에 Encoder를 사용하여, Embedding Space에 Mapping했을 때, 네트워크 내에서의 두 노드의 인접성이 Embedding Space에서도 유지되어야 하는 것입니다. 즉, Embedding을 하기 위해선 다음 두 개념이 필요합니다.

* 노드 $u$를 Embedding Space의 $z_u$로 Embedding하는 Encoder
* Network 내에서의 $u,v$의 유사성이 벡터의 유사성을 나타내는 내적값 $z_v^Tz_u$와 유사함을 나타내는 Similarity Function이 필요하다. 

**"Shallow" Encoding**

Encoder를 단순히 Embedding Look-up으로 간주하는 것이 Shallow Encoding입니다. 

![image-20210114214607253](/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114214607253.png)

마치 Word2Vec에서 등장했던 Lookup table과 같이 $v$는 단순히 Indicator Vector로서, 모든 Embedding이 담겨있는 행렬 $Z$에서 어떤 Column을 가져올 지 지정하는 것입니다. 이것이 가장 단순한 기법이지만, 이 후에 Deep Walk / Node2Vec / TransE와 같은 다양한 알고리즘을 배워볼 것입니다. 

# Random Walk Approaches to Node Embeddings

Random Walk란 이름이 나타내듯, 그래프 내를 '임의로 걷는' 기법입니다. 

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114093622143.png" alt="image-20210114093622143" style="zoom:67%;" />

임의의 노드, 위의 그림에서는 4번 노드에서 그래프를 임의의 방식으로 순회하여 도달하게 되는 지점까지를 Node Sequence로 나타냈을 때, 해당 Sequence가 Random Walk가 되는 것입니다. 특히 하나의 노드에서만 출발하는 것을 [Random Walk with Restart]라고 하지만 Random Node에서 출발하는 것도 가능하며, 몇 개의 노드를 거치느냐와 같은 파라미터로 Random Walk를 구성할 수 있습니다.

**그럼 Embedding과 Random Walk랑은 무슨 연관이 있을까요?**

> $z_u^Tz_v \approx$ Random Walk 내에 Node $u,v$가 공존할 확률

위와 같은 식으로 두 노드의 Similarity를 나타내는 '내적'을 Random Walk에 대한 식으로 연결시킬 수 있습니다. 즉,

![image-20210114094650385](/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114094650385.png)

1. (좌측 그림) 노드 $u$에서 Random Walk Strategy를 이용해 순회할 때, $v$를 방문할 확률입니다. 이는 사실 $v$가 도착지점일 필요는 없습니다.
2. 두 벡터의 내적을 표현하는 $cos(\theta)$와 Random Walk확률 $P_R(v|u)$를 일치시키는 방식으로 훈련을 진행합니다. 

**왜 굳이 Random Walk인가?**

* Local과 High order 이웃 정보까지 반영한 Node 유사도를 확률 표현으로 정의한다.
* 유사도를 계산하기 위하여 Input으로 모든 노드를 사용하지 않고, Random Walk에 등장한 Pair만을 사용한다.

결과적으로 Random Walk는 하나의 노드가 모든 노드를 고려하는 대신 인접한 주위 노드만 계산에 포함하므로, 주위의 노드와의 관계를 잘 보존하는 Node Embedding을 생성하는 과정에서 비용적 / 의미적 측면에서 적합하게 됩니다. 

> $N_R(u)$: 특정한 Random Walk 전략 R에 의해 획득한 $u$의 이웃

해당 방법으로 Optimization을 진행하는 방식은 다음과 같습니다.

$G = (V, E)$로 표현되는 그래프가 있을 때, $z: u \rightarrow R^d$, $u$를 $d-dimension$ vector z로 만드는 과정에서 Log-likelihood를 사용합니다.

> $max_{z}\sum\limits_{u\in V}logP(N_R(u)\|z_u)$ :  
> 이웃에 Random walk로 인해 나타난 노드들이 포함되어 있을 확률을 최대화 하는 Node Embedding $z_u$를 찾아라!

1. 그래프의 모든 각 노드에 대하여 R 전략으로 short fixed-length random walk를 시행한다.
2. 각 노드 $u$에 대하여 중복을 허용하는 $N_R(u)$를 생성한다. 이 때 중복 집합인 이유는 Random Walk로 인해 하나의 노드를 여러 번 방문할 수 있기 때문이다.
3. $u$가 주어졌을 때, 이웃들 $N_R(u)$를 예측한다.

위의 Optimization Function이 현재 확률의 최대화지만, 보통 Minimization으로 목적식을 전환하기 때문에 음수를 붙이고 Softmax에 대한 식으로 바꾸면 최종식이 나타납니다.

![image-20210114153641088](/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114153641088.png)

이 때, Softmax 식을 사용하는 이유는 모든 노드들에 대한 비교를 진행해야 하기  때문이며, 그 중 값이 가장 커야 Loss Function이 작아지기 때문입니다. 의미적으로 합당한 것이 어차피 $v$는 $u$에서 시작한 Random Walk의 결과물에 포함되어 있기 때문에 값이 높을 것이기 때문입니다. 

## Negative Sampling for Random Walk Opt.

하지만 위의 방식에서는 계산 복잡도가 매우 커지게 됩니다. 

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114155034058.png" alt="image-20210114155034058" style="zoom:50%;" />

위의 Loss Term에서 모든 Vertex를 순회하는 것은 반드시 필요한 과정이지만 Softmax를 Normalize하기 위한 분모 항으로 인해 또 다시 Vertex만큼의 계산이 필요하게 되며, 이른 Nested Sum은 결국 $O(\|V\|^2)$로 나타나게 됩니다. 따라서 이런 복잡한 식에 대하여 근사를 실시하는 것이 좋습니다.

근사를 위해 Softmax Function을 Maximize하는 대신 Negative Sampling을 사용하는 방식으로 전환합니다.

![image-20210114161128032](/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114161128032.png)

위의 방식으로 전환하게 되는데, $\approx$ 우측에 있는 식 중 좌항은 목표 노드 $u$와 이웃노드 $v$를 뜻하며, 우항의 $n_i$는 $k$개의 Negative Sampling을 의미합니다. 

* **Negative Sampling**
  Word2Vec에서 사용되는 Negative Sampling은 모든 단어에 대한 Update를 하는 대신, 중심 단어와 상관 없는 단어들을 사용하여 Update한다. 이 때 등장 빈도수가 높은 단어들은 범용적으로 활용되는 경우가 많기에 특정 단어와의 관계가 비교적 크지 않을 것이기에 그런 단어들에 가중치를 주어 Sampling한다.
* **Noise Contrastive Estimation(NCE)**
  Softmax의 Log 확률을 근사하는 방법으로서, Negative Sampling을 아우르는 기법

또한 알아두셔야 할 것이, 

* 우항의 값은 일반적으로 좌항보다는 작을 것이며, 
* $n_i$는 Random Distribution이기 때문에 $v$를 추출해버릴 수도 있다는 것,
* 그리고 sigmoid function이 확률의 값으로 치환한다.
* Negative Sampling 개수 $k$는 노드의 degree에 따라 다르다.

해당 부분에 대하여 저희 이탈리아인 교수는, 더 궁금한 것이 있으면 [Word2Vec Explained](https://arxiv.org/abs/1402.3722)를 살펴보라고 합니다.

강의에서 다음과 같은 질문이 나왔었습니다.

**Question**: 그럼 그냥 주위 Neighborhood로서 계산하면 될 것이지 뭣하러 Random Walk를 포함하냐?

**Answer**: 두 가지 이유가 있는데,

* 첫번째: Efficiency를 달성할 수 있는데, 그 이유는 하나의 노드의 이웃이 엄청 많다고 생각해봐라.
* 두번째: 1-hop / 2-hop만을 사용하는 것은 전체 그래프의 의미를 함께 반영할 수가 없다. 하지만 Random Walk는 어떤 전략을 사용하느냐에 따라서 Deep한 관계까지 포함하여 Long-term Dependency를 학습할 수 있다.

## Deep-Walk의 문제점

결국 이런 식으로 Random Walk를 통해 Embedding을 정의하는 것이 Deep-Walk였습니다. 이는 Random Walk의 개념을 그대로 사용해서 적용했기에 가장 간단한 Idea인데,

* Fixed-Length Random Walk $\Rightarrow$ 같은 role을 갖는 Node가 멀리 떨어져 있다면 상호관계를 파악할 수 없다.
* Unbiased Random Walk (Random / Uniform)

와 같은 특징을 갖고 있습니다. 하지만 이는 지나치게 제한적인 기법이라 일반화를 진행해야 합니다. 그것이 바로 Node2Vec의 탄생 배경입니다.

# Node2Vec

기본적으로 Deep-Walk와 Node2Vec의 지향점은 같습니다. 

* Feature Space 내에서 비슷한 노드 이웃들은 비슷한 값으로 Embedding해야하며, 
* Maximum Likelihood Optimization을 진행하며,
* Downstream Prediction Task에 Independent 해야합니다. 

이를 위해 Node2Vec에서는 Biased $2^{nd}$ order random walk를 진행하여 $u$의 이웃 $N_R(u)$를 구성하게 됩니다. 즉 Random Walk를 진행하는 방식이 달라 이웃 정의법이 다르지만, 그 이후의 절차는 같습니다.

**선이수과목**

먼저 사전 지식으로 그래프의 순회방식인 BFS / DFS를 살펴보겠습니다. 

* DFS, Depth First Search / 깊이우선탐색
* BFS, Breadth First Search / 넓이우선탐색

해당 개념들은 GNN 관련 내용에서만 존재하는 개념이 아니라, 알고리즘 등 다양한 저변에서 사용되는 개념이므로 설명은 하지 않겠습니다. 일반적인 알고리즘의 방식은 아래와 같습니다.

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/img.gif" alt="img" style="zoom:50%;" />

하나의 그래프에서 시작 노드를 $u$라고 할 때, 길이 3의 Walk, 즉 이웃 노드 사이즈가 3이라고 할 때, BFS를 거친 뒤의 이웃 노드, DFS를 거친 뒤의 이웃노드는 다릅니다. BFS를 Local Microscopic View라고 명명한 이유는 결국 자기 주위의 근접 Hop을 먼저 다방면으로 살펴보기 때문이며, DFS를 Global Macroscopic View라고 명명한 이유는 한 방향으로만 깊게 파고들어 멀리멀리 뻗어나가기 때문입니다. 

<img src="/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114202858056.png" alt="image-20210114202858056" style="zoom:50%;" />

다음으로 먼저 알아두어야 할 개념은, 'Biased Random Walk'입니다. Deep-Walk는 어떤 Edge들에 대해서도 편견없이 완전 Random으로 노드를 Traverse했습니다. 하지만 Node2Vec에서는 방향마다 정해진 확률이 존재하여 특정 방향으로 확률이 더 높은 Biased Random Walk를 진행합니다. 

* **Return Parameter $p$:**
  자신이 출발했던 노드로 돌아가는 확률의 파라미터
  Node2Vec의 알고리즘이 $2^{nd}$ Order인 이유는 바로 자신이 왔던 곳을 기억하고 있기 때문이다.

* **In-out Parameter $q$:**
  새로운 모험을 떠난다(DFS, Moving Outward) vs 다시 내 고향으로 돌아간다 (BFS, Inwards)
  따라서 $q$는 BFS와 DFS의 비율을 정해주는 파라미터다.

  ![image-20210114203920888](/assets/2021-01-15-Graph-Representation-Learning.assets/image-20210114203920888.png)

하나의 노드 $s_1$에서 $w$로 이동을 한 상태라고 가정을 해보겠습니다. 그렇다면 해당 노드가 취할 수 있는 행동은 세 가지가 되는데, 

* 첫 번째, DFS를 통해 $s_1$에서 멀어지려고 한다.
* 두 번째, BFS를 통해 고향의 품, $s_1$로 돌아오려고 한다.
* 세 번째, $s_2$로 이동하는데 이 때 이는 DFS가 아닌 것이 $s_1$이 기준점이기에 거리가 동일하기 때문이다. 즉, $s_1 \rightarrow w$와 $s_1 \rightarrow s_2$는 똑같은 거리이기 때문에 이동을 했다고 할 수 없기 때문이다.

와 같은 선택을 할 수 있습니다. 

$p$는 출발점으로 회귀하는 파라미터이므로, Low value이면 BFS의 행동을 취하며, $q$는 출발점에서 멀어지려는 파라미터이므로, Low Value이면 DFS의 행동을 취하는 것입니다. 위와 같은 절차로 움직였을 때 나타나는 $N_R(u)$로 인해 biased walk를 거치게 됩니다.

따라서 최종적으로 Node2Vec 알고리즘을 정리하면 다음과 같습니다.

1. Biased $2^{nd}$ order random walk 확률을 정의한다.
2. Node $u$에서 $r$번의 random walk를 $l$의 길이만큼 진행한다.
3. Node2vec Objective를 Stochastic Gradient Descent로 진행한다.

