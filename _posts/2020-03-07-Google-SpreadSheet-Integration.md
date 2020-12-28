---
layout: post
title: "Gspread & Google SpreadSheet"
categories: [crawling, GCP, project]
---
# 크롤링 데이터, gspread => Google SpreadSheet

데이터를 Scrape를 해온 후, 다음의 행동은 여러 갈래로 나뉜다. 데이터를 메일로 전송할 지, 데이터베이스에 전송할 지, 또는 엑셀로 계속 추출할 것인지 등 말이다. Scrapy를 사용할 때의 장점은 "-o output.csv" 파라미터를 넣어주면 해당 csv에서 맨 마지막 row에 지속적으로 데이터를 업데이트 해준다. 하지만 결국 해당 절차는 csv를 또 옮겨서 별도 절차를 거쳐야 한다.

따라서 나는 이를 보완하기 위하여 google SpreadSheet에 바로 데이터를 전송할 수 있도록 설정하고자 한다. 이를 통해 나를 포함한 고객이나 다른 사용자들이 데이터의 움직임을 실시간으로 Tracking이 가능해진다.

해당 절차는 매우 간단하다. 뭔가 복잡한 설정이라기보다 하라는거 하고, 깔라는거 깔면 된다.

매우 잘 정리되어 있는, https://yurimkoo.github.io/python/2019/07/20/link-with-googlesheets-for-Python.html 를 참고하여 작업하였다.
