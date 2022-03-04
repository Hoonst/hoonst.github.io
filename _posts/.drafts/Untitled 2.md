# UDA Plan of Attack
1. Model: BERT Classification
2. 무조건 Back Translation 적용하고
3. Labeled Data는 우리가 Label 단 것
4. Unlabeled Data는 그 이외의 데이터


# 형석
GNN의 여러 Task 중 두 그래프가 동일한 형태를 갖고 있는지(Isomorphic)를 판단하는 과제가 있으며 이는 Weisfeiler-Lehman Test가 있습니다. 이런 Test로 두 그래프의 동형성을 판단할 수 있지만 일부의 경우에서 완벽한 정답을 내놓지 않는 것이 단점입니다. 확실히 다른 그래프를 구분하지 못하는 것은 Graph 자체의 Expressiveness가 부족하다는 뜻이며 본 세미나에선 이를 풍부하기 위한 두 가지 논문을 소개하였습니다. GNN에서 개별 그래프의 표현력이 감소하는 이유는 단훈히 이웃의 정보들을 취합하는 MPNN 때문이며, DropGNN 같은 경우는 Dropout node를 통해 매번 다른 perturbated variants를 통해 학습하여 그래프 내의 수많은 Subgraph를 생성하여 표현력 향상을 목표로 합니다. Dropout과 같은 방식으로 Subgraph를 구성하는 첫번째 논문과 다르게 두 번째 논문에서는 DS / DSS로 대표되는 Subgraph encoding을 통해 GNN 표현력 향상을 달성했으며, Siam Network 구조를 통해 Permutation된 각 subgraph 정보를 공유하는 Layer를 구성합니다. 그래프의 관심이 떨어질때마다 충전해주시는 세미나 해주셔서 감사합니다. 