문서 요약의 평가를 위해 사용되는 ROUGE Score를 살펴보기 위하여 다음과 같은 예시를 살펴보겠습니다.

* Reference Summary: 정답 요약문으로서, 사람이 직접 구성한 요약문이며 모델의 결과가 해당 요약문을 참조해 자신이 요약문을 잘 생성해냈는 지 평가하는 참조(Reference)의 역할을 합니다.

  **ex: the cat was under the bed**

* Candidate Summary: 요약 모델이 Return한 Output으로서의 요약문으로서, 정답의 후보(Candidate)가 되는 요약입니다.

  **ex: the cat was found under the bed**

해당 Summary를 통해 Precision과 Recall 을 계산하는 방식은 다음과 같습니다.

Recall: ![image-20201230110458780](C:\Users\yoonh\AppData\Roaming\Typora\typora-user-images\image-20201230110458780.png)

Reference Summary 의 단어는 5개이고, Candidate & Reference Summary의 겹치는 단어들은 5개이므로, $5/5 = 1$이 Recall의 결과가 됩니다.

Precision: ![image-20201230110658711](C:\Users\yoonh\AppData\Roaming\Typora\typora-user-images\image-20201230110658711.png)

Candidate Summary의 단어는 6개이고, Candidate & Reference Summary의 겹치는 단어들은 5개이므로, $5/6$이 Precision의 결과가 됩니다.

ROUGE Score는 Recall과 Precision을 사용하지만, n-gram을 사용하여 평가를 더 Robust하게 진행합니다. 이 때, n-gram이란, n개의 연속적인 단어 나열을 뜻하며, n은 사용자가 직접 설정하는 Hyperparameter로써, n을 2로 설정하면 bi-gram, 3으로 설정하면 tri-gram이 됩니다. 예시로 위의 Reference Summary, Candidate Summary로 Bi-gram(n=2)을 구성해보겠습니다.

Reference Summary: the cat was under the bed

Bi-gram: (the, cat) / (cat, was) / (was, under) / (under, the) / (the, bed)

Candidate Summary: the cat was found under the bed

Bi-gram: (the, cat) / (cat, was) / (was, found) / (found, under) / (under, the) / (the, bed)

해당하는 Bi-gram을 통해 ROUGE-2(Bi-gram Overlap)를 계산하게 되면

ROUGE-2(Recall): $4/5$

ROUGE-2(Precision): $4/6$

로 계산되게 됩니다.

ROUGE-L은 Longest common subsequence를 사용하여 계산하게 됩니다. Longest common subsequence란 연속적이지 않아도 되나 순서가 같은 Sequence를 뜻하며 위의 예시에서는 'the cat was under the bed'가 됩니다. 따라서 Recall / Precision 을 계산하게 된다면, 

ROUGE-L(Recall): $5/5$

ROUGE-L(Precision): $5/6$

으로 나타낼 수 있습니다.

 





