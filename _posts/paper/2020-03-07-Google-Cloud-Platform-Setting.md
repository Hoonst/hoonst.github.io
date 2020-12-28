---
layout: post
title: "GCP Setting"
categories: [crawling, GCP, project]
---
# GCP Settings for Scheduling Scraper
본 Setting을 진행하면서 확실히 구글이 위대한 기업이지만, 튜토리얼만큼은 기깔나게 못만드는 구나 싶었다. 물론 내가 GCP에 대한 지식이 크지 않아서 헤맸을수도 있지만... 너무 불친절하다. 뭔가 친절하게 불친절한 느낌?

[문제의 gcloud compute 예약 튜토리얼](https://cloud.google.com/scheduler/docs/start-and-stop-compute-engine-instances-on-a-schedule)

먼저 내가 구현하려고 했던 계획은 다음과 같다.
1. 매일 24:00에 전날 올라온 QnA List에 대한 Scraping을 하는 Scrapy 제작
2. 클라우드에 올려 24:00에 해당 Scrapy가 작동하게 하는 '.sh' 파일을 제작
3. 클라우드가 23:50에 켜지고 1:00에 닫도록 설정
4. Scraping한 데이터를 Gspread를 통해 SpreadSheet에 전송

위에서 나는 튜토리얼이 불친절하다고 했다. 하지만 사실 엄밀하게 말하자면, '틀린' 튜토리얼이었다. 내가 모르는 별도의 설정이 있었을 지는 모르겠지만, 똑같은 코드와 방법으로 튜토리얼을 따라했지만 작동하지 않았다. 이에 StackOverFlow에 직접 질문을 하여 답변을 받았고(차후에 설명) 천천히 문제를 풀어나갔다.


### Compute Engine 인스턴스 설정
```
gcloud compute instances create workday-instance \
    --network default \
    --zone asia-northeast2-a
```

### Cloud Pub/Sub로 Cloud Functions 함수 설정

```
gcloud pubsub topics create start-instance-event

gcloud pubsub topics create stop-instance-event
```

그 이후 구글이 만들어 놓은 github에서 디렉토리를 clone 해온다.

```
git clone https://github.com/GoogleCloudPlatform/nodejs-docs-samples.git
```
또는 [zip파일]("https://github.com/GoogleCloudPlatform/nodejs-docs-samples/archive/master.zip")을 통해 다운 받을 수 있다.

### 시작 중지 함수

그리고 시작 및 중지함수를 만들기 위하여 **nodejs-docs-samples/functions/scheduleinstance/** 디렉토리 안에 있어야 한다.

```
cd nodejs-docs-samples/functions/scheduleinstance/
```
**첫번째 고난의 등장**

이때 우리는 첫번째 오류에 봉착할 것이다.
```
gcloud functions deploy startInstancePubSub \
    --trigger-topic start-instance-event \
    --runtime nodejs6

gcloud functions deploy stopInstancePubSub \
        --trigger-topic stop-instance-event \
        --runtime nodejs6
```
위 코드에서 nodejs6 부분을 그대로 둔다면 에러가 나타날 것이다. 뭐가 없다 뭐가 없다... 계속 이러는데 구글링의 결과, nodejs6 버전은 deprecated 됐다고 한다. 그런데 왜 수정하지 않고 그대로 냅뒀는지 모르겠다. 따라서 해당 부분을 nodejs8 또는 10으로 설정하고 진행해야 한다.

###(선택사항) 함수 작동 확인  
**두번째 고난의 등장**

함수가 잘 설정되어 있고 이를 확인하기 위해서 function을 활용해서 실제로 Compute Engine이 켜지고 꺼지는 실험을 해봐야한다. 사실 선택사항이기 때문에 이것이 안된다고 전체 Flow가 작동이 안되는 지는 모르겠다. 하지만 뭔가 그냥 넘어가기에는 꺼림칙하기에 진행을 했으며, 결국 거의 하루 남짓의 시간을 사용하였다.

1. 인스턴스를 중지하는 함수를 호출한다.
```
gcloud functions call stopInstancePubSub \
    --data '{"data":"eyJ6b25lIjoidXMtd2VzdDEtYiIsImluc3RhbmNlIjoid29ya2RheS1pbnN0YW5jZSJ9Cg=="}'
```
여기서 위의 'data' key의 value인 조잡한 문자열은
`{"zone":"us-west1-b", "instance":"workday-instance"}`
를 base64로 인코딩한 문자열이다. 따라서 사실 인코딩이라는 것이 '인코딩=정보' 이고 표현하는 방법만 다르기 때문에 같은 input값이라고 생각하여 위와 같은 인코딩이 문제가 있자 그냥 raw하게 넣어보았으나 이 역시 에러를 불러일으킨다. 따라서 StackOverFlow에 질문을 해서 원인과 결과를 얻어내었다.

[Life Saver, StackOverFlow]("https://stackoverflow.com/questions/60500583/gcp-scheduler-error-function-execution-failed-details-attribute-label-missi?noredirect=1#comment107097109_60500583")

본 질문을 작성하고 얼마 후 답변이 올라왔다.
![QA](../img/comments_stackoverflow.png)

위의 문제는 자꾸 'label'이 missing한다고 에러를 내뱉는 현상이었다. 애초에 파라미터에 label이 없는데 그게 없다니 너무 어이가 없었다. 

![error](../img/error_message_stack.png)
그리고 답변도 당연하긴 했지만, 당혹스러웠다. 당연히 label이 없으니 label을 넣으라는 뜻은, 마치

"1+1은 3이야"  
"왜?"  
"사실 1+1에다가 +1하는 것은 당연한거니까 3이야"

이런 느낌이었다.

따라서 해결을 위해 'label'에 대한 정보는 전무했지만 그냥 하라는 대로  
`{"zone":"asia-northeast2-a", "label": "env=dev", "instance":"workday-instace"}`  
를 base64로 encoding하고 'data'에 넣었다. 이로서 정상 작동 메시지가 나타났으며 문제를 해결하였다. 현재 댓글에 GCP 직원이 직접 문제에 대한 해결책을 적어달라고 부탁하는 칸이 있다. 뭔가 신기하면서도, "애초에 잘하지 그랬냐..." 라는 느낌이 들기도 한다.

![success](../img/success_message.png)
