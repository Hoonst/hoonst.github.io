---
layout: post
title: "Crontab의 에러 극복 노트"
excerpt: "Crontab, 자꾸 생각나는구나"
categories: [GCP, Crontab]
image:
  feature: https://images.unsplash.com/photo-1519052537078-e6302a4968d4?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1500&q=80
---
# Crontab, 넌 대체 뭐니...

GCP를 사용한 프로젝트에서 Compute Engine을 Cloud Scheduler로 깨워, Web Crawler를 Crontab을 통해 작동시켜 매일매일 작동하게 설정해둔 적이 있다. 그런데 어느날부터 다음과 같은 현상이 일어났다.

![compromised](../img/compromised_compute_engine.png)

갑자기 나타난 빨간 느낌표의 compute engine은 가상화폐 채굴을 하는 해킹(?)당한 엔진이고 이는 결국 내 프로젝트를 날려버려, 처음부터 다시 프로젝트를 설정하게 되었다.

이때 git에 저장을 해놨기 때문에 전체적일 프로그램은 문제가 없었다. 하지만 문제는 .sh 파일과 crontab은 미처 git에 올려놓지 않아 다시 설정을 하는데 이상하게 이전에 겪은 일이지만 기억이 나지 않아 애먹고 있다.

이는 Live Problem Solving으로서 한번 한 솔루션씩 진행해보겠다.

문제 상황:  
1) `crontab`을 보편적인 방법으로 실행해보았지만 활성화되지 않음.  
2) cron화 시키자하는 .sh 파일은 단독 실행시 잘 돌아감
3) 지난 번에는 `timezone`의 문제로써 Ubuntu내의 시간은 UTC timezone인데 나는 한국 시간으로 설정했기 때문에 작동하지 않았기에, 이를 수정하여 문제를 풀었다. 하지만 이번에는 이것이 되지 않는다.

잊지 말아야 하는 가정:  
`내 코드는 문제없다`

# Preliminary Test:

먼저 실험을 해야할 것이 있다. 내 코드는 완벽하게 돌아갔지만 과연, cron 자체는 지금 돌아가기는 하는 것일까? 시간에 상관없이 매분마다 결과를 볼 수 있는 코드를 통해 실험해보
```
* * * * * env > /tmp/env.output
```
env을 찍게되면 내 컴퓨터에 대한 여러가지 설정이 나오며 `>`를 통해 `/tmp/env.output` 안에 결과를 넣는다.

다행히도 결과가 env.output안에 출력이 된다.
즉 crontab 자체에는 문제가 없다는 뜻이다.

그럼 뭐가 문제일까...

사실 이전에 여러 검색을 통해 시도한 것은 주로,
* Ubuntu .sh file은 /bin/sh가 default라 #!/bin/bash, 전체 커맨드 앞에 sh 붙이기 등 다양한 시도가 필요하다.  
라는 조언을 따라해보았다. 하지만 소용이 없었다.

따라서 이것의 로그를 관찰해봐야겠다는 생각이 들었다. 로그를 살펴보는 방법은 여러가지가 있지만 그 중에서도 crontab은 default로 메일을 사용자에게 보내 현재 cron이 어떻게 돌아가는 지 알려주는 것을 생각해내어 해당 폴더로 들어가보았다.

```
cron 완료 후 나타나는:

You have new mail in /var/mail/yoonhoonsang
```
해당 폴더의 파일을 살펴보면,

![mailbox](../img/mailbox.png)

즉 가상환경이 문제가 있어서 그 어떤 패키지도 import하지 못하고 전체 코드가 돌아가지 않음을 보인다. Google이 내 프로젝트를 삭제하기 이전의 프로젝트에서는 venv를 사용해서 단순하게
```
source scraper/bin/activate
```
를 하게 된다면 가상환경을 작동하는데에 문제가 없었다. 하지만 pyenv를 사용하는터라 환경변수를 명확하게 설정해야했다. 이를 위하여
```
export PATH="${HOME}/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi
```
를 ~/.bashrc에 삽입하였다. 하지만 이 역시도 Fail!

따라서 또 다시 이곳저곳 돌아다니다가 결국 발견한 것은...sh file에 아래를 첨부하는 것이다.
```
source ~/.profile
```
뭐 설명에 따르면 결국 pyenv의 위치를 제대로 알려준다는 꼴인데 완벽하게 이해하지는 못했다...
여하튼 다시 프로젝트 복구 완료!

crontab final
```
00 00 * * * /bin/bash /home/yoonhoonsang/internet_lecture/execute_all_scrapers.sh
```
