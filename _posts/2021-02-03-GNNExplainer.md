---
₩	₩	₩₩layout: post
title: "GNN Explainer"
description: "Rex Ying et al.(2019) "
tags: [paper_review]
date: 2021-02-03
comments: true
typora-root-url: ../../hoonst.github.io
---

# Introduction

GNN의 장점에 대해서 논할 때 항상 등장하는 말이 '관계성'입니다. Non-relational Data, 즉 우리가 일반적으로 살펴볼 수 있는 DataFrame 꼴의 데이터는 각 데이터들이 Independent하다는 가정을 갖고 있는데, 이는 Social Network, Chemical Data 등과 같이 관계가 중요한 요소로 자리매김 하는 Relation Data과 차이가 있으며 이에 대해선 해당 관계성을 반영할 수 있는 기법을 사용해서 분석해야 합니다. GNN이 바로 이 관계성을 담을 수 있는 기법이므로, 얽히고 설킴이 강한 현대 사회의 네트워크를 분석하기에 최적의 기법인 것입니다.

GNN이 관계성을 정의하는 방식은 대표적으로 Adjacency Matrix입니다. Laplacian, Degree Matrix 등과 같이 다른 행렬들도 존재하지만, 실질적으로 보면 Adjacency가 근간을 이룹니다. Adjacency Matrix를 통해 하나의 노드 주위에 어떤 노드들이 존재하는 지, 즉 이웃을 파악할 수 있고 Graph 전체의 구조도 파악할 수 있게 됩니다. 

위와 같은 방식의 GNN은 일반적인 Neural Network보다 성능이 크게 상승하지만, 그 복잡성 역시 상승하여 Black Box로 치부되던 딥러닝이 더 어두워지는 느낌이 있습니다. 따라서 Explainability를 달성하기가 어렵지만 이를 달성해야만 하는 이유는

* GNN Model에 대한 신뢰성 / 투명성 향상
* Network 특징을 파악함으로써 모델에 존재할 수 있는 Error를 파악

이 가능하기 때문입니다. 

Graph의 Explainability를 다룬 논문은 신기하게도 이전에는 하나도 존재하지 않아, 본 논문이 최초라고 할 수 있으며 비슷한 시기에,

> Explainability Methods for Graph Convolutional Neural Networks (Phillip E. Pope et al / CVPR, 2019)

가 나오긴 했지만 간발의 차로 더 빠릅니다. 또한 해당 논문은 GCNN 계열의 기법들에만 적용할 수 있는 방법으로서, 기존의 CNN이 Saliency Map을 구성하기 위해 존재하던 기법을 그대로 가져다 사용한 것이지만 GNNExplainer는 최적화 방식으로 진행합니다.

# GNN Explainer

GNN Explainer를 GNN의 Prediction을 설명하기 위한 기법으로서, 훈련된 GNN과 Prediction을 가지고 특정 노드나 Class에 대하여 Subgraph를 구성하며, 이를 통해 어떤 노드가 하나의 노드 또는 Class를 구성하는데 영향을 주었는지 알 수 있습니다. 또한 영향 노드만을 구할 수 있는 것이 아니라, 노드의 어떤 Feature의 영향력이 큰지에 대해서도 알 수가 있어 더 심화적인 분석이 가능합니다. 정리하자면

* 하나의 노드를 구성함에 있어 어떤 이웃의 노드들이 영향을 미쳤는지 (Single-instance explanations)
* 하나의 Class를 구성함에 있어 어떤 이웃의 노드들이 영향을 미쳤는지 (Multi-instance explanations)
* 노드 내의 어떤 Feature가 영향을 미쳤는지

로 나누어볼 수 있습니다. 

<img src="/assets/2021-02-03-GNNExplainer.assets/image-20210203160629361.png" alt="image-20210203160629361" style="zoom:67%;" />

위의 그래프를 예시로 살펴보겠습니다. 각 노드는 학생들이며, Label은 취미 운동을 어떤 것을 갖고 있는지에 대한 그래프를 나타내고 있습니다. $\Phi$를 Node Classifcation을 진행하는 GNN Model이라고 했을 때, $v_i, v_j$ 노드가 어떤 운동을 선택할 지 판단하는 문제라고 할 수 있는데, 빨간색의 모델이 $v_i$를 농구라고 분류하고 초록색의 모델이 $v_j$를 Sailing이라고 분류하는 경우입니다. 훈련된 모델과 Prediction이 존재할 때 GNNExplainer는 각 노드가 판단을 하기 위한 요소들이 무엇인지 판단하여 우측의 그림처럼 Subgraph를 나타낸 것을 볼 수 있습니다. 빨간색 모델과 농구로 분류한 경우를 살펴보면 $v_i$노드는 자전거, 달리기와도 연결이 되어있지만 '공과 관련된' 운동들이 더 주요한 영향을 끼쳤기에 농구라고 분류했다고 할 수 있습니다.

이를 달성하기 위하여 GNN Explainer는 GNN의 Prediction과 Mutual Information을 최대화 하는 Subgraph를 탐색합니다. 

## Related Work

**Explainability 기법 종류**

GNN의 Explainability를 다룬 선례가 없기 때문에 기존 ML / DL에서 사용했던 방식을 정리하자면 크게 두 가지 방식으로 나뉩니다.

* Surrogate / Proxy Model을 구성한 뒤, Explanation을 찾는다.
* High level feature를 설명할 수 있는 연관 변수를 찾는다.
* Computation 과정 속에서 주요한 변수를 찾는다.
  $\Rightarrow$ Feature Gradients, Backpropagation, Counterfactual reasoning
  이런 기법들로 생성한 Saliency Map은 잘못된 결과를 낳을 수 있고, Gradient Saturation과 같은 문제를 겪을 수 있는데, 특히 Adjacency Matrix에서 문제가 증폭될 수 있기에 GNN에 사용하기에는 부적합한 기법이다.

하지만 이런 기법들은 Graph의 정수라고 할 수 있는 관계성을 포함하지 못하기 때문에, Node Feature와 관계 정보를 모두 담을 수 있는 기법이 필요하게 됩니다.

Post-hoc interpretability methods

Attention

## Formulating explanations for graph neural networks

먼저 본 논문에서 정의하는 Notation을 정리하고, GNN의 절차를 간략하게 정리해보겠습니다.

* G: Graph
* E: Edge
* V: 

## GNNExplainer: Problem Formulation

<img src="/assets/2021-02-03-GNNExplainer.assets/image-20210210133234973.png" alt="image-20210210133234973" style="zoom:67%;" />

GNN 기법으로 Node $v$의 Embedding $z$를 판단하고자 합니다.

* $G_c(v)$: $v$를 $z$ Embedding으로 구성하는 Computation Graph. Computation Graph는 계산 과정을 위해 필요한 노드들을 엮은 그래프를 뜻한다.
  위의 그림 A에서 색깔로 표시된 그래프들이 Computation Graph이며, 초록색은 $\hat y$를 구성할 때 중요한 노드들이며, 노란색은 반대의 의미이다.
* $A_c(v) \in \{0,1\}^{n \times n}$: Adjacency Matrix는 0과 1사이의 값으로 $n \times n$ 차원으로 이루어져 있다.
* $X_c(v) = \{x_j\|v_j \in G_c(v)\}$: $v$를 구성하기 위한 Computation Graph내에 있는 노드들의 Feature들이다. 
* $P_{\Phi}(Y\|G_c, X_c)$: 노드들의 Feature들과 Computation Graph를 사용했을 때 {1, ..., C} 종류의 Label을 가질 확률을 표현한 것이다. 

여기까지가 기존 모델에 대한 Notation 입니다. 위의 Notation을 통해 모델을 구성하고 그에 따른 결과들의 확률을 표현할 수 있습니다. 이 중, GNNExplainer는 일부의 Subgraph와 일부의 Sub-Feature를 Masking을 통해 얻어내고자 합니다.

* $G_S$: Computation Graph $G_c(v)$의 Subgraph
* $X_S$: $G_S$에 포함되어 있는 Feature
* $X_S^F$: $X_S$ 중 일부의 Features / 위의 그림 B에서 빨간색 x로 삭제되지 않은 Features이다.



## 본격 GNNExplainer

GNN Explainer를 위한 준비물:

* $\Phi$: Trained GNN Model
* Prediction (Single Instance에 대한 Explanation이 필요하면 하나의 Prediction이 필요하고, Multi-Instance에 대한 설명이라면 Predictions)

## Single Instance Explanations

Single Instance에 대한 Explanation을 얻는 과정은 다음과 같습니다.

* Node $v$가 주어졌을 때, 해당 노드를 구성하기 위한 전체 Computation Graph 내에서 Subgraph를 탐색 $\Rightarrow$ $G_S \subseteq G_c$

* SubGraph의 Feature들인 $X_S = \{x_j\|v_j \in G_S\}$를 탐색

GNN Explainer는 위의 과정으로 목적을 달성하기 위하여 Mutual Information(MI)을 통해 식을 구성하며, 목적식은 다음과 같습니다.

> $max_{G_S}MI(Y, (G_S, X_S)) = H(Y) - H(Y\|G = G_S, X = X_S)$

MI는 머신러닝에서 사용되는 가장 기초적인 개념이며, Cross Entropy 등 다양한 개념의 근간이기도 합니다. 하지만 기초라고 하기엔 은근히 이해에 난이도가 있지만, 간단하게 표현하자면

> 하나의 무작위 변수가 변할 때, 다른 변수에 대해 얻어지는 '정보량'을 나타내는 것

이라고 할 수 있습니다. 

즉, MI가 GNN Explainer에서 사용되는 것은 풀어서 말하자면, $G_S, X_S$가 변함에 따라 $Y$가 얼마나 변하는 지를 나타내는 정보량이며, GNN Explainer는 그 수치를 최대화하는 것이 목적입니다. 논문 그대로의 표현을 차용하자면, MI는 $\hat y = \Phi(G_c, X_c)$의 확률이 $G_S, X_S$로 제한되었을 때 얼마나 변하는지를 나타내는 것입니다. 

MI의 식을 보면 $H(Y)$와 $H(Y\|G = G_S, X = X_S)$의 차로 나타내는데, 후자의 값이 작아야 MI를 최대로 만들 수 있습니다. 즉, $H(Y\|G = G_S, X = X_S)$의 값이 작아야 한다는 것인데 이는 $G_S / X_S$를 사용할 때, $Y$의 Uncertainty 또는 분산이 적다는 뜻이므로 좋은 예측 변수 Set이라고 말할 수 있습니다. 

$G_S$의 개수를 지정해야 Compact한 Explanation을 구성할 수 있기 때문에, $K_M$보다 작도록 구성하며 ($\|G_S\| \leq K_M$) 이를 통해 $G_c$를 Denoising해 핵심 $K_M$개의 노드를 구할 수 있습니다. 

**본격 GNN Explainer Optimization**

하나의 노드를 구성하는 $G_c$가 크기가 꽤 커, 10개의 노드로 구성된 Subgraph라고 한다면, 이 10개로 구성할 수 있는 Subgraph는 매우 클 것입니다. 





## Joint Learning of Graph Structural and Node Feature Information

GNN Explainer는 단순히 어떤 노드 조합이 Embedding에 큰 영향을 미치느냐만 조사하는 것이 아니라, 어떤 Feature가 우수한지도 살핍니다. 이는 단순히 위의 MI 식에서 $G_S$ 즉, 그래프 구조만이 식에 포함되어 있는 것에 반해 $X_S^F$를 통해 Feature의 영향력도 살펴보고자 하는 것입니다. 따라서 식에 약간의 추가만을 진행합니다.

> $max_{G_S , F}MI(Y, (G_S, F)) = H(Y) - H(Y\|G = G_S, X = X_S^F)$

추가된 항목은 $F$로서, Feature Mask의 역할을 하며 이를 학습해야 합니다. 만약 특정 변수가 중요하지 않다면 어차피 학습을 통해 Node Embedding 구성이 적은 영향을 주도록 가중치가 작아질 것입니다. 하지만 만약 큰 역할을 갖고 있다면 가중치 행렬 내의 값이 커질 것이므로 그것을 $F$ Feature Mask로 가려버린다는 것은 예측 확률의 저하를 불러올 것입니다.



## GNN Explainer Model Extensions

* **그래프를 활용한 어떤 Task도 가능하다.**
* **어떤 GNN 모델이어도 가능하다.**
* **계산 복잡도**

## Experiments

XAI류의 모델들에서 성능 평가는 생소합니다. 과연 각 모델들은 어떻게 성능을 평가하여 이전 기법들과의 비교를 통해 자신들의 기법의 우수성을 자랑할 수 있을까요? Accuracy를 기반으로 하는 것 같지만, 예측 값이 위에서 쭉 설명해온 Subgraph나 Feature라고 했을 때 Label이 무엇인지 잘 모르겠습니다. 따라서 실험 섹션을 보면서 사용하는 데이터 셋과 실험 방법을 살펴보겠습니다. 보통 실험은 저자들의 자랑 시간이라고 생각하여 주의 깊게 살펴보지는 않았지만 요즘 들어 Experiments를 잘 읽어야 논문 전체가 이해 가능한 경우가 많다고 생각이 듭니다.

### Datasets

**Synthetic datasets**

<img src="/assets/2021-02-03-GNNExplainer.assets/image-20210210155815943.png" alt="image-20210210155815943" style="zoom:67%;" />

* BA-Shapes
  * 300개의 노드를 가진 Barabasi-Albert(BA) Graph에서 80개의, 5개 노드로 구성된 House 꼴의 Motif를 Random으로 연결한다.
  * 0.1N (N = Node 개수)개의 Edge를 추가적으로 포함해 Perturbation
  * 그래프 내 각 노드들은 4개의 Class가 존재: House의 머리/ 가슴 / 배, House에 포함되지 않는 노드
* BA-Community
  * 2개의 BA-Shapes의 합집합이며, Community와 Role에 따라 8개의 Class로 나누어진다.
* Tree-Cycles
  * 8층의 Tree를 구성한 뒤에 80개의, 6개 노드로 이루어진 Cycle을 가져다 붙힌다.
* Tree-Grid
  * 8층의 Tree를 구성한 뒤에 80개의, 9개 노드로 이루어진 (3x3) Grid를 붙힌다. 

**Real-World Datasets**

* MUTAG
  * 4,337 분자 그래프이며, Gram-Negative Bacterium S.typhimurium으로 인한 유전적 변화로 Labeled
* REDDIT-BINARY
  * 2,000개의 그래프는 각 Discussion Thread를 나타낸다. 
  * Node: User In Thread / Edge: 하나의 유저가 다른 유저에 답장했을 경우
  * Graph는 Thread내에 어떤 User Interaction이 이루어져 있는지 판단한다. 
    * Question-Answer Interaction: $r/IAmA, r/AskReddit$
    * Online-Discussion Interactions: $r/TrollXChromosomes, r/athesim$

**Alternative Baseline Approaches**

GNN Explainer와 다르게 기존의 기법들은 Graph에 직접적으로 적용하지 못합니다. 하지만 비교 군을 수립하기 위하여 몇가지 기법들로 논문에서는 비교를 하나 구체적인 방법에 대해서는 기술하지 않아 구현 코드를 살펴봐야 할 것 같습니다.

* GRAD
  * GNN Loss Function의 Gradient를 Adjacency Matrix에서 계산해내는데 Saliency Map을 구성하는 것과 동일하다.
* ATT
  * GAT를 통해서 Computation Graph의 Edge들에 대하여 가중치를 계산할 수 있는데 Feature에 대한 가중치는 알 수 없다. 

**Setup and implementation details**

* 각 데이터 셋에 대하여 먼저 GNN을 적용하여 훈련
* GRAD / GNN Explainer를 통해 GNN의 예측값을 설명
* ATT는 GAT를 통해 Attention Weight를 구성
* $K_M, K_F$는 각기 Subgraph와 Feature Explanation의 Size를 나타낸다.
* Synthetic Dataset에서는 $K_M$을 Ground Truth로 설정
* Real Dataset에서는 $K_M = 10$으로 나타내며, $K_F$는 모든 데이터 셋에서 5



실험을 통해서 저자들이 밝혀내고자 한 질문들은 다음과 같습니다.

* GNN Explainer, 꽤 괜찮은 설명을 할 수 있는가?
* Ground-Truth Knowledge와 Explanation을 어떻게 비교하는가?
* Graph 기반 예측 Task에서 GNN Explainer를 어떻게 사용하는가?
* 훈련된 GNN말고 다른 GNN에도 바로 적용할 수 있는가? (Inductive)



**Quantitative Analyses**

![image-20210210164852273](/assets/2021-02-03-GNNExplainer.assets/image-20210210164852273.png)

Node Classification Dataset에 대한 Accuracy를 위의 표에서 확인할 수 있습니다. GNN Explainer를 통해 예측한 값은 무엇이고 Label은 무엇일까요?

* Predicted: Explainability Method으로 예측해낸 Importance Weights
* Label: Ground-truth Explanation에 포함되어 있는 Edge

즉, Ground-Truth Label을 예측해나가는 Binary Classification으로 문제를 정의해 나갔습니다. 더 우수한 Explainability Method는 더 정확히 Ground Truth Edge들을 예측해냈기에 Accuracy가 높게 나타납니다. 



**Qualitative Analyses**

![image-20210210172207069](/assets/2021-02-03-GNNExplainer.assets/image-20210210172207069.png)

Synthetic Datasets의 Task를 다시 한번 살펴보자면, 먼저 노드들에는 4개의 Class가 있으며 해당 Class들을 맞추는 것이 목적입니다. 그리고 기본적으로 Base Graph에 연결되어 있는 Motif가 Explanation, 또는 Ground Truth라고 정의하고 있으며 이는 Figure 3에서 나타내고 있습니다. GNN Explainer는 정확하게 House / Cycle / Tree Motif들을 잘 예측해내어 하나의 노드를 설명하기 위한 Explanation을 탐지했으나, Grad나 Att 기법들은 제각각의 결과를 나타내는 것을 볼 수 있습니다.

Real World Dataset에서는 Graph Classification Task를 진행하는데, 하나의 전체 그래프를 판별하기 위해 필요한 Motif들을 Return 합니다. 따라서 GNN Explainer로 MUTAG 데이터셋에 적용했을 때, 각 Graph를 분류하기 위한 특정 Motif들을 잘 나타내지만, 다른 기법들은 Ground Truth와 거리가 좀 멀어보입니다.

REDDIT-BINARY Dataset에서는 Label이 각 그래프가 어떤 유형의 thread인지를 판단하는 목표를 갖고 있었습니다. QA thread를 보게 되면 2-3개의 높은 Degree의 노드가 다른 낮은 Degree의 노드와 많이 연결되어 있는 것을 볼 수 있습니다. 생각해보면, Reddit과 같은 커뮤니티에서 답을 하는 사람들은 소수의 똑똑한 사람들이기 때문이기에 일리가 있어보입니다. 따라서 그래프 자체의 Label을 판별하는 과정에서 어떤 Subgraph를 획득하는 과정을 통해 모델 자체를 설명할 수 있게 되는 것입니다. 

**확실하게 해야 할 것**

모든 Graph Based [Model]은 목적이 있습니다. 그리고 이 목적은 Node Classifcation / Graph Classifcation 등 다양하게 나타날 것입니다. 하지만 헷갈리는 것이 있는데 이는 Ground Truth입니다. 

* Synthetic Dataset $\Rightarrow$ Base Graph에 연결된 Motif

* Real-World Datasets $\Rightarrow$

  * MUTAG: 분자 구조를 구성하는 하위 요소를 다 알고 있으니 Ground Truth를 지정 가능
  * REDDIT: 각 thread의 일반적인 특징을 생각해서 정의

  



