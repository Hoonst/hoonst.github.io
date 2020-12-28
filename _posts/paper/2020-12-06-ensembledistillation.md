---
layout: post
title: "Ensemble Distillation for Neural Machine Translation"
description: "Markus Freitag, Yaser Al-Onaizan, Baskaran Sankaran (arXiv 2017)"
date: 2020-12-06
categories: paper
tags: [review, code, deeplearning, distillation]
image: 
---


# [1] 아이디어 제안 배경

## Knowledge Distillation
Knowledge Distillation(지식 증류; KD)은 2015년 G.Hinton에 의해 제안된 모델 경량화 방법론입니다. 모델의 크기를 키워 좋은 성능을 내도록 하는 것은 연구의 중요한 목표 중 하나이지만, 모델 배포 측면에서 본다면 분명히 개선해야 할 사항 중 하나입니다. KD는 모델을 배포하기에 적당한 크기의 모델이 약간의 도움을 받아 기존보다 더 좋은 성능을 내도록 하는 방법론입니다.

### Ensemble Model
일반적으로 모델의 성능을 높이는 방법은 앙상블 기법을 적용하는 것입니다. 즉, 같은 데이터에 대해 여러 모델을 학습한 후 각각의 예측값에 평균을 취하여 최종 예측을 하면 성능이 좋아진다는 것입니다(여기서는 앙상블 모델의 출력을 개별 모형 출력의 평균으로 정의하겟습니다). 직관적으로 이해가 잘 되는 개념이지만, 수식을 통하여 앙상블 기법의 효과에 대해 알아보겠습니다. 이 부분은 [고려대학교 강필성 교수님의 강의](https://youtu.be/mZwszY3kQBg)를 참고하였습니다.


먼저 용어에 대해 아래와 같이 정의하도록 하겠습니다.
  
- True function: $f(\mathbf{x})$
- Estimation of $m$-th model: $y_m(\mathbf{x}) = f(\mathbf{x})+\epsilon_m(\mathbf{x})$

따라서 오차에 대한 기댓값은 아래와 같이 정의할 수 있습니다. 

$$ {\mathbb{E}_x[\{y_m(\mathbf{x}) - f(\mathbf{x})\}^2] = \mathbb{E}_\mathbf{x}[\epsilon_m(\mathbf{x})^2]} $$

비교할 값은 아래와 같습니다.

1. 단순히 $M$개 모델의 오차 기댓값을 평균

$$ E_{Avg} =  {1\over M} \sum_{m=1}^M \mathbb{E}_{\mathbf{x}} \left[\epsilon_m(\mathbf{x})^2 \right] $$

2. $M$개 모델에 앙상블 기법을 적용한 후 오차 기댓값

$$ E_{Ensemble} = \mathbb{E}_\mathbf{x} \left[\left\{ {1\over M} \sum_{m=1}^M y_m(\mathbf{x}) - f(\mathbf{x}) \right\}^2 \right] $$

$$  = \mathbb{E}_\mathbf{x} \left[\left\{ {1\over M} \sum_{m=1}^M y_m(\mathbf{x}) - {1\over M} \sum_{m=1}^M f(\mathbf{x}) \right\}^2 \right] $$

$$  = \mathbb{E}_\mathbf{x} \left[ \left\{ {1\over M} \sum_{m=1}^M \epsilon_m(\mathbf{x})  \right\}^2 \right] $$


여기서 한 가지 비현실적인 가정을 하겠습니다. 첫 번째로는 **오차들의 평균이 0**이라는 가정과, 두 번째로는 **오차들이 서로 uncorrelated 관계**라는 가정입니다. 식으로는 다음과 같이 표현할 수 있습니다.

$${\mathbb{E}_\mathbf{x} [\epsilon_m(\mathbf{x})] = 0}$$ 

$${\mathbb{E}_\mathbf{x} [\epsilon_m(\mathbf{x}) \epsilon_l(\mathbf{x})] = 0 \;\;(m \neq l)}$$

이 상황에서 ${E_{Avg}}$와 ${E_{Ensemble}}$의 관계는 아래와 같이 표현할 수 있습니다. 이는 앙상블 모델의 오차 기댓값이 개별 모형의 오차에 모형의 개수를 나눈 만큼까지 작아질 수 있음을 의미합니다(correlated한 경우에는 ${1\over M}$이 아니라 공분산계수가 포함된 값이 등장합니다). 

$$ E_{Ensemble} = {1 \over M} E_{Avg} $$

그러나 오차가 correlated한 현실적인 경우에서는 [코시-슈바르츠 부등식](https://ko.wikipedia.org/wiki/%EC%BD%94%EC%8B%9C-%EC%8A%88%EB%B0%94%EB%A5%B4%EC%B8%A0_%EB%B6%80%EB%93%B1%EC%8B%9D)을 사용하여 아래와 같이 표현할 수 있습니다.

$$ \left[ \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right]^2 \le (1^2+1^2+\cdots+1^2) \sum_{m=1}^M \epsilon_m(\mathbf{x})^2 $$

$$ \left[ \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right]^2 \le M \sum_{m=1}^M \epsilon_m(\mathbf{x})^2 $$

$$ \left[ {1\over M} \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right]^2 \le {1 \over M} \sum_{m=1}^M \epsilon_m(\mathbf{x})^2 $$

$$ E_{Ensemble} \le E_{Avg} $$

정리하자면, 현실적인 상황에서도 앙상블 모델은 개별 모델보다 낮은 오차를 가진다는 것입니다.

### Teacher to Student
앙상블 모델에 대한 설명이 길어졌는데, 요지는 모델의 성능을 높이기 위해서 앙상블 기법을 많이 사용한다는 것입니다. 하지만 여러 모델이 필요한 만큼 그에 비례하여 연산량이 증가하게 됩니다. KD 논문에서는 이 문제를 지적하며 앙상블 모델의 성능을 작은 크기의 단일 모델에 전달하는 방법을 제시합니다. 


다시 표현하자면, distillation은 큰 모델의 지식을 작은 모델에게 가르쳐 주는 것입니다. 따라서 큰 모델을 Teacher 모델, 작은 모델을 Student 모델이라 부릅니다. 여기서 Teacher 모델의 지식은 softmax 이후의 출력값을 의미합니다. 모델이 분류해야 하는 클래스가 있다면, Teacher 모델은 많은 파라미터를 학습하여 어떠한 입력이 주어졌을 때 각 클래스로 분류하는 확률(Teacher 모델의 지식)을 출력하게 되는 것과 같습니다.


다만, softmax 확률값이 특정 클래스를 너무 확실하게(1에 가깝게) 예측한다면 Teacher 모델은 확실하게 예측하는 클래스 외의 정보는 거의 갖지 못합니다. 따라서 softmax에 "Temperature"라는 개념을 도입하여 분포를 완만하게 만든 후 Student 모델에 전달하게 됩니다. 데이터의 양이나 특성에 따라 적당한 Temperature가 각각 다르기 때문에 적당한 값을 찾을 필요가 있습니다.


# [2] 방법론
본 논문에서는 신경망 기반 기계번역(Neural Machine Translation; NMT)에 distillation을 적용합니다. Teacher 모델에서 Student 모델로 전달하는 '지식'과 '방법'은 시간이 흐르면서 굉장히 다양해졌습니다. 또한 기존 KD가 주로 vision task에 적용되었던 반면 본 논문에서는 번역 모델에 적용하는 것처럼 범용적으로 활용되고 있습니다. 논문에서는 크게 세 가지의 distillation 방법을 제안합니다.


## 1. Ensemble Teacher Model
일반적인 Teacher 모델은 Student 모델보다 큽니다. 본 논문에서는 거대한 Teacher 모델 대신, Student와 같은 크기의 Teacher 모델을 사용합니다. 다만 여러 개의 Teacher를 앙상블하여 단일 Teacher 모델보다 좋은 성능을 내도록 합니다(디코딩 속도가 느려진다는 단점은 존재합니다). 앙상블된 Teacher 모델을 Student 모델로 distill하는 목적은 더 적은 연산량과 더 빠른 속도로 앙상블 모델의 성능을 얻는 것입니다. 구체적인 동작은 디코딩 시 매 time step마다, 랜덤한 값으로 초기화된 각 모델이 출력하는 확률값을 평균하는 방식으로 이루어집니다.

## 2. Oracle BLEU Teacher Model
모델이 디코더를 거쳐 출력할 때에는 beam search를 많이 사용합니다. Beam search는 각 time step마다 log probability가 가장 높은 $k$개의 후보를 저장합니다. 그리고 문장이 종료된 이후 가장 높은 log probability를 갖는 시퀀스를 선택하고, 정답 번역문과의 loss를 계산합니다.


그러나 본 논문에서는 이미 정답 번역문을 알고 있는 상황을 활용하여, 가장 높은 log probability를 갖는 시퀀스를 선택하는 대신, 정답 문장과의 BLEU Score가 가장 높은 시퀀스를 선택하는 모델도 사용합니다.

## 3. Data Filtering Method
기계번역에서 입력문-출력문의 쌍은 대부분 웹 크롤링을 통해 구축합니다. 또한 하나의 문장에 대해 여러 가지로 번역할 여지가 존재합니다. 이러한 이유로 잘못된 번역쌍이나 하나의 문장에 여러 번역이 가능한 경우 모델의 학습이 어려워집니다. 따라서 본 논문에서는 학습된 Teacher 모델로 번역한 데이터(forward translation)를 구축하고, 실제 번역문과의 TER(Translation Error Rate) Score를 계산합니다. TER Score는 출력한 번역문을 실제 번역문과 비교하여 post-editing이 얼마나 되었는지를 측정하는 0~1 사이의 측도입니다. TER Score가 낮을 수록 번역의 질이 높다고 간주하고, 논문에서는 높은 TER Score를 갖는 문장들을 삭제하는 방법을 사용하였습니다. Distillation 이전에 이러한 전처리를 거칠 경우 더 빠른 학습과 성능 향상의 효과를 가져온다고 합니다.


# [3] 실험
실험에 사용한 데이터는 아래와 같습니다.

- Train: WMT 2016 (German → English)
- Valid: newstest2014
- Test: newstest2015

더하여 자체적으로 변경한 attention과 byte-pair encoding을 사용합니다. 그리고 같은 실험을 두 번씩 진행하는데요, 차이점은 아래와 같습니다.

1. 랜덤하게 초기화된 가중치에서 처음부터 학습
2. 학습된 Baseline(단일 Student 모델)의 파라미터에서 추가로 학습

구현 코드는 공개하지 않았습니다. 보다 자세한 설정은 생략하고 학습 결과를 보면 아래와 같습니다.


## Single Teacher Model

<img src="/assets/figures/endis_t1.PNG" width="70%">

앙상블을 적용하지 않은 단일 Teacher 모델에 대해 distillation을 수행한 결과입니다.
Teacher의 번역(forward translation)만 사용하면 효과가 없지만, 정답 번역문과 함께 활용하는 경우 더 나은 결과를 얻을 수 있었습니다. 또한 일정 TER를 기준으로 데이터를 필터링한 경우 성능 자체는 엇비슷하지만 12%정도의 속도 향상 효과가 있었다고 합니다.

## Ensemble Teacher Model

<img src="/assets/figures/endis_t2.PNG" width="70%">

6개의 Teacher 모델을 앙상블하여 distillation을 적용한 결과입니다. Forward translation만 사용한 Student 모델도 단일 Teacher 모델보다는 높은 성능을 보여줍니다. 역시 정답 번역문을 함께 사용하고 데이터 필터링을 수행한 경우 더 높은 성능 향상을 보입니다.

## Oracle BLEU Teacher Model

<img src="/assets/figures/endis_t3.PNG" width="70%">

앙상블한 Teacher 모델 구조는 그대로 유지한 채, beam search에서 log probability가 아닌 BLEU Score를 기준으로 최종 출력값을 선택한 모델에 대한 성능입니다. 단일 Student 모델에 비해 인한 성능 향상은 있었으나, Teacher 모델에 비해서는 성능이 많이 떨어지는 것을 확인할 수 있습니다.

## Reducing Model Size

<img src="/assets/figures/endis_t4.PNG" width="70%">

Student 모델은 Teacher 모델과 동일한 구조를 갖지만, 토큰 임베딩과 신경망의 은닉층 크기를 줄여서 distillation을 진행했습니다. 은닉층 크기가 300, 토큰 임베딩 크기가 150인 경우가 가장 작은 상황인데, 파라미터 수를 상당히 줄였음에도 불구하고 단일 Student 모델보다 성능이 높아졌음을 확인할 수 있습니다. 파라미터 수를 적당히 줄인 경우에는 앙상블한 Teacher 모델과도 크게 차이가 나지는 않습니다.


다만 forward translation을 사용한 것이 어떻게 보면 augmentation 효과를 갖는다고 볼 수도 있는데, 그렇다면 데이터 수가 다른 모델 성능 비교가 합당한 지에 대해서는 생각해 보아야 할 것 같습니다.


# [4] 코드

먼저 양해를 구하자면(~~핑계를 대자면~~), 신경망 기반의 기계번역 모델은 학습하는 데 굉장히 많은 시간이 필요했기 때문에 결과까지 도출하지는 못하였습니다. 데이터 수가 워낙 많기도 하고 논문의 설정을 그대로 따라하기에는 장비의 한계가 있어 최대한 줄였음에도 학습 시간이 굉장히 오래 걸립니다. 제 장비(2080ti)로만 학습을 시키면, beam search를 사용하지 않았음에도 앙상블 모델은 커녕 Teacher 모델 하나 학습하는 데에도 너무 많은 시간이 걸립니다. 얼른 transformer 기반 모델로 넘어가야 할 필요성을 절실히 느낍니다.


기본적으로 Bahdanau Attention 논문과 유사한 구조로 번역모델을 설계합니다. 따라서 인코더, 어텐션, 디코더 클래스를 포함한 Sequence-to-Sequence 모델입니다. 코드는 아래와 같습니다. 전체 코드는 [여기](https://github.com/youngerous/ensemble-distillation)에서 확인하실 수 있고, 특징적인 몇 개의 구조에 대해서만 이 글에 적어두겠습니다.

## Seq2Seq

논문의 구현처럼 앙상블 기법을 적용하기 위해서는 여러 개의 모델을 함께 초기화해야 하는데, 조그마한 장비에서는 하나의 모델도 겨우 올릴 수 있었습니다. 

```python
class Seq2Seq(pl.LightningModule):
    """
    In order to apply ensemble, multiple encoder and decoder should be initialized.
    But only one encoder and decoder exist because of hardware limitation.
    At each decoding time step in ensemble situation,
      multiple decoder outputs(probabilities) are averaged.

    :param input_dim: src vocabulary size
    :param output_dim: tgt vocabulary size
    :param vocab: byte-pair encoding vocab
    :param emb_dim: embedding size
    :param hid_dim: rnn hidden size
    :param max_len: maximum length of sentence
    :param lr: learning rate
    :param teacher_force: teacher forcing ratio
    :param kd: whether to apply knowledge distillation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        vocab: dict,
        emb_dim: int,
        hid_dim: int,
        max_len: int,
        lr: float,
        teacher_force: float,
        kd: bool = False,
    ):
        super(Seq2Seq, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            input_dim=self.hparams.input_dim,
            emb_dim=self.hparams.emb_dim,
            hid_dim=self.hparams.hid_dim,
            bidirectional=True,
        )
        self.decoder = Decoder(
            output_dim=self.hparams.output_dim,
            emb_dim=self.hparams.emb_dim,
            hid_dim=self.hparams.hid_dim,
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=1)

    def _create_mask(self, src):
        mask = src != self.hparams.vocab["<PAD>"]
        return mask

    def forward(self, srclen, tgtlen, src, tgt, teacher_force=0.0):
        # srclen = [batch size]
        # tgtlen = [batch size]
        # src = [batch size, max len]
        # tgt = [batch size, max len]
        batch_size = tgt.shape[0]
        outputs = torch.zeros(self.hparams.max_len, batch_size, len(self.hparams.vocab)).to(
            self.device
        )

        enc_outputs, (hidden, cell) = self.encoder(srclen, src)
        _input = tgt[:, 0]  # <SOS>
        mask = self._create_mask(src)

        for t in range(1, self.hparams.max_len):
            output, hidden, cell, _ = self.decoder(_input, enc_outputs, hidden, cell, mask)
            outputs[t] = output

            teacher_force = random.random() < self.hparams.teacher_force
            top1 = output.argmax(1)  # greedy decoding
            _input = tgt[t] if teacher_force else top1

        return outputs
```

## Distillation 
Student 모델을 학습할 때 이미 학습된 Teacher 모델을 불러와서 output을 가져옵니다. 그 후 아래의 loss function으로 Student 모델을 학습합니다. 이 역시 Teacher 모델을 불러오는 과정에서 메모리가 터졌습니다. 이후의 디버깅이 불가능해졌기 때문에 구현은 distillation 코드까지로 마쳤습니다.

```python
class SoftTarget(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    ref: https://github.com/AberHu/Knowledge-Distillation-Zoo
    """

    def __init__(self, T=1):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(
                F.log_softmax(out_s / self.T, dim=1),
                F.softmax(out_t / self.T, dim=1),
                reduction="batchmean",
            )
            * self.T
            * self.T
        )

        return loss
```

<img src="/assets/figures/endis_error.PNG" width="70%">

> 마지막의 느낌표가 정말 얄미웠습니다.


# [5] 마치며

2017년 arXiv에만 올라온 논문이고 발견 당시 인용수가 약 50회였던 것을 보면, NMT 분야에서 범용적으로 활용하기 좋은 기술들을 많이 사용했기 때문이지 않나 생각이 듭니다. 본 논문의 기여점을 요약하면 아래와 같습니다.

- Teacher 모델에 앙상블 기법을 적용하여 실험하였음
- Forward translation과 정답 요약문 모두를 사용한 실험과 더불어, forward translation만 사용하는 경우에 대해서도 실험하였음 (그리고 두 개를 모두 사용한 것이 좋음을 확인하였음)
- 데이터 필터링을 진행하는 것이 속도를 빠르게 하고 성능도 향상시키는 것을 확인하였음
- Teacher와 Student의 크기가 같아도 효과가 있음을 확인하였음
- Student 모델을 처음부터 학습하는 경우와, 학습된 baseline Student 모델의 파라미터에 이어서 학습하는 경우에 대한 비교를 진행하였음


한편으로는 본 논문의 실험 자체는 성능 향상이 있었지만 distillation과 앙상블을 함께 사용하는 것은 trade-off를 종합적으로 고려해보아야 한다는 생각이 들었습니다. Distillation 자체도 거대한 Teacher 모델을 학습한 후 Student에 전달하는 특징을 갖고 있기 때문에 cost가 높은데, 여기에 앙상블까지 더하면 향상되는 성능에 비해 cost가 지나치게 많이 소모되는 경우가 있으리라 생각합니다. 본 논문에서 다루었던 큰 데이터와 큰 구조를 지닌 번역모델을 서비스화하는 경우가 대표적인 예시가 될 것 같습니다.

# [6] 참고자료
- [[Paper] Ensemble Distillation for Neural Machine Translation](https://arxiv.org/abs/1702.01802)
- [[Paper] Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
- [[Paper] Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy](https://www.researchgate.net/publication/220344230_Measures_of_Diversity_in_Classifier_Ensembles_and_Their_Relationship_with_the_Ensemble_Accuracy)
- [[Post] What is Translation Error Rate(TER)?](https://kantanmtblog.com/2015/07/28/what-is-translation-error-rate-ter/)
- [[YouTube] Ensemble Learning - Bias-Variance Decomposition](https://youtu.be/mZwszY3kQBg)