---
layout: post
title: "CS224W Lecture 2"
description: "Properties of Networks, Random Graph Models"
tags: [CS224W]
date: 2021-01-06
comments: true
typora-root-url: ../../hoonst.github.io
---

CS224W 2번째 강의 정리를 정리해보았습니다. 강의에서 사용된 이미지 전체는 CS224W 수업 자료에서 가져왔음을 미리 밝히는 바입니다. 

2번째 강의에서는 Network의 특징을 표현하는 방법, 그리고 Graph를 표현하는 모델에 대하여 살펴보겠습니다.



# Network Properties

하나의 Network을 파악하고 다른 Network와 구분할 수 있는 특징들은 다음과 같습니다. 지금부터 설명하는 특징들은 Undirected Graph를 대상으로 나타내는 것이며, Directed Graph에 대해 특별히 설명할 경우 언급하겠습니다.

* **Degree Distribution**, $P(k)$ : Random하게 선택한 Node가 degree $k$를 가질 확률입니다. Degree는 하나의 노드에 몇개의 Edge가 달려있느냐를 표현합니다. 

  $N_k$ = 하나의 노드의 Degree $k$

  $P(k) = N_k /N$: Normalized Histogram

  Directed Graph같은 경우에는 in- / out-degree distribution이 따로 존재합니다.

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106175313762.png" alt="image-20210106175313762" style="zoom:50%;" />

* Path: Path는 한 노드에서 다른 노드로 도착하기 위해서 거쳐야 하는 노드들의 Sequence를 말합니다. 다음과 같은 그래프가 있을 때, A와 G의 Path를 계산한다면, 여러 Path가 존재하겠지만 ACBDCDEG와 같이 나타낼 수 있고, 이는 같은 노드를 여러 번 방문할 수 있음을 알 수 있습니다.

![image-20210106175824857](/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106175824857.png)



* **Distance**

Path 중에서 가장 짧은 Path를 Distance라고 정의하며, 만일 연결되어 있지 않다면 0이나 무한대로 나타냅니다. 

비방향성 그래프는 Distance가 $h_{A,B} / h_{B,A}$가 같지만, 방향성 그래프는 다를 수 있기에 Distance가 대칭의 꼴이 아닙니다.

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106180453074.png" alt="image-20210106180453074" style="zoom:67%;" />

​		**Diameter**

​		Diameter는 Distance(shortest path) 중에서 가장 큰 Distance입니다. 최단 거리 중 가장 크다는 뜻은 전체 그래프 사이즈와도 연관이 되기에 Diameter라고 표현한 듯 싶습니다.		

**Average Path Length**

Diameter는 최대 Distance를 나타낸 것과 다르게, Average Path Length는 단순히 Distance의 평균을 나타내는 것입니다.

* Clustering Coefficient (for undirected graphs)

  $i$ 번째 노드의 Clustering Coef.란 $i$의 이웃끼리의 연결의 강도를 나타냅니다.

  $C_i = \frac {2e_i} {k_i(k_i -1)}$로 나타내게 되며 

  * $e_i$가 이웃간에 연결된 edge 개수이며, 
  * $k_i$는 이웃의 개수입니다.

  따라서 $k_i(k_i-1)$는 $k_i$개의 노드가 서로를 연결하는 Edge의 최대 개수를 말합니다. N개의 노드에 대하여 Average clustering coefficient는 $C = \frac {1}{N} \sum\limits_i^N c_i$입니다.

  ![image-20210106182013893](/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106182013893.png)

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106182147756.png" alt="image-20210106182147756" style="zoom:67%;" />

* Connectivity

  Connectivity는 '연결'의 정도를 나타내며, Largest Connected Component라 함은 가장 많은 노드들이 연결되어 있는 그래프라고 할 수 있습니다. 아래의 그림과 같이, ABCD가 묶여 있는 Component가 가장 크다고 할 수 있습니다.

  <img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106183041943.png" alt="image-20210106183041943" style="zoom: 67%;" />

# MSN Messenger

지금까지 살펴본 Network의 큰 특징 4가지를 살펴보았습니다. 다음으로는 실제 세계의 Network에서 해당 특징들이 어떻게 나타나는지 살펴볼 것입니다. 원래 이런 예시들은 단순히 예시로만 남는 경우가 있는데, 이번 예시는 뒤의 개념과 이어지기에 언급하고 넘어가도록 하겠습니다.

강의에서 사용하는 예시는 MSN Messenger 사용자들에 대한 예시입니다. 

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106183343951.png" alt="image-20210106183343951" style="zoom:67%;" />

다음과 같이 사람들이 누구에게 메시지를 보내고 대화를 했는지를 Edge로 나타내며, 해당 Network를 구성할 때를 기점으로 한 달 사이에 1개의 메시지라도 보냈으면 Edge로 연결했습니다. 그리고 180 Million의 사람들이 Node로서 나타나고 1.3 billion edge들이 존재합니다. 

## Degree Distribution

![image-20210106183817310](/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106183817310.png)

위의 그래프는 Node의 Degree별 Count를 나타낸 그래프입니다. **좌측 그래프**를 살펴보면 Degree가 어림잡아 0~200 정도의 Degree를 가진 노드의 빈도가 가장 높으며 그 이상으로 넘어갈 수록 0에 수렵합니다. **우측 그래프**는 각 축을 로그 변환하여 나타낸 것인데, 해당 그래프가 Degree가 높아질수록 기하급수적으로 빈도가 낮아지는 양태를 더 잘 보여주고 있습니다.

# Erdos-Renyi Random Graph Model

위에서 보여드린 예시는 '실제 세계'의 Network입니다. 따라서 어떤 System의 Network를 구성하느냐에 따라 Degree Dist. 등과 같은 특징의 양태가 달라질 것입니다. 이에 '보편적으로는' Graph가 어떤 특징을 갖게 될 지에 대하여 모델을 구축한 것이 Random Graph Model이며, 그 중 대표적인 Erdos-Renyi의 모델을 살펴보겠습니다.

해당 모델을 구성하는 데에는 두 가지 Variant가 존재합니다.

* $G_{np}$: undirected graph에서 n개의 노드가 존재하고, 각 edge는 확률 p로 생성된다.
* $G_{nm}$: undirected graph에서 n개의 노드가 존재하고, m개의 edge가 uniform / random으로 생성된다.

강의에서는 $G_{np}$를 통해 모델이 어떤 그래프를 생성해내고, 특징은 어떻게 나타나는지 살펴보았습니다.

본격적으로 특징을 보여드리기에 앞서, n과 p가 동일하더라도 다양한 그래프가 나올 수 있음을 기억하셔야 합니다. n개의 노드는 fix되어있다고 해도 연결을 담당하는 edge가 확률값이기 때문에, 어떤 그래프에서는 두 노드 $(u,v)$가 연결이 되어있을 수도 있고 아닐 수도 있기 때문입니다.

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106185703755.png" alt="image-20210106185703755" style="zoom:67%;" />

## Degree Distribution

**Fact: $G_{np}$의 Degree Distribution은 Binomial Distribution이다.**

Random Graph Model에서 두 노드가 연결될 확률은 p로 나타납니다. 따라서, 두 노드가 연결이 되냐, 연결이 안되냐 라는 이항분포로 나타낼 수 있습니다. 하나의 노드가 Degree $k$를 갖는 확률은 아래의 식 같이 나타낼 수 있습니다.

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106190556115.png" alt="image-20210106190556115" style="zoom:67%;" />

이항 분포의 식으로 확률 분포를 나타낼 수 있고, 따라서 분포의 기대값과 분산을 구해낼 수 있습니다.

위의 그림의 우측 하단을 살펴보면, 분산을 평균에 대한 함수로 나타내는 것을 보실 수 있습니다. 그리고 해당 함수를 전개하게 되면 분모가 $(n-1)$로 나타나게 되는데, 이는 노드 개수가 증가할수록 그래프의 분산이 작아진다는 것을 의미합니다. 분포의 분산이 작아진다는 뜻은 모든 값들이 $k$의 평균값에 근사한다는 말과 동치가 되게 됩니다.

## Clustering Coefficient of $G_{np}$

저희는 위에서 $C_i = \frac {2e_i} {k_i(k_i -1)}$ 와 같이 Clustering Coefficient를 나타내는 것을 볼 수 있었습니다. 

 <img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106192113947.png" alt="image-20210106192113947" style="zoom:67%;" />

이때, 분자에 있는 $e_i$는 $i$ 노드의 이웃들간의 edge 개수를 나타내는데, 따라서 이 또한 이항분포의 기대값으로 나타낸다면, $np$의 꼴로 나타내야 하며, $n$ 이 edge의 경우 edge 개수가 $\frac {k_i(k_i-1)} {2}$로 나타낼 수 있기에 위의 식과 같게 됩니다. 그 이후 전체 $C_i$에 대한 식으로 나타내게 되면

<img src="/assets/2021-01-06-Properties-of-Networks-Random-Graph-Models.assets/image-20210106192442377.png" alt="image-20210106192442377" style="zoom:67%;" />

위의 식으로 나타낼 수 있습니다. 해당 식이 나오는 이유는 $k_i$가 고정된 상수이기 때문에 기대값 식 밖으로 빠져나오기 때문이라고 생각하시면 이해가 되실겁니다. 

앞에서 degree distribution에 대하여 설명드릴 때, 노드 개수가 증가할수록 분포의 분산이 줄어 모든 노드가 degree $k$를 가질 것이라고 했습니다. 즉, $E[C_i] = \frac {\bar k}{n}$으로 표현될 때, 그래프의 크기가 커질수록 $k$의 평균은 고정일테지만 분모인 노드 개수는 커질 것이며 이는 기대값이 작아지게 만들 것입니다. 따라서 Clustering Coefficient는 감소하여, 하나의 노드의 Neighbor들끼리의 연결 강도가 낮아질 것입니다.

## Path Length of $G_{np}$



## Connected components of $G_{np}$



# The Small-World Model

