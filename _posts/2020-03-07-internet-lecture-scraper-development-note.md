---
layout: post
title: "Internet_Lecture_Scraping_Project"
categories: [crawling, project]
---
# Internet_Lecture_Scraping_Project
##### 인터넷 강의(인강) 강사 QNA 갯수 크롤링 자동화

이번 대학원 및 취업 준비를 하면서, 무언가 실질적으로 개발을 하고 싶다는 생각과 프로젝틀르 진행하고 싶다는 생각이 들었다. 따라서 고민하던 중, ETOOS 윤훈관 강사(영어 will-be 1타 강사)의 제안으로 경쟁자들의 현황을 지속적으로 파악할 수 있는 시스템을 만들어보고자 했다.

인터넷 강사들의 인기 지표는 단순히 '수강생' 만으로 표현하기에는 부족하다. 얼마나 수강생들과 소통하는 지, 그리고 그 강의에 대한 평이 좋은지에 대한 여러가지 복합적인 요인을 조사할 필요가 있다. 그 중, 'QnA', 즉 질문답변이 얼마나 활발하게 이루어지는 지 살펴보면, 해당 강사의 '활동성'을 파악할 수 있으며, 이는 성공의 지표로도 활용할 수 있다.

따라서 본 프로젝트에서 달성하고자 하는 목표는 다음과 같다.

1. 대표 영어 인강 강사 설정:  
  * 이투스: 윤훈관 / 강원우
  * 메가스터디: 조정식
  * 스카이에듀: 전홍철
  * 대성마이맥: 이명학   링크 걸어두기
2. Scrapy로 Scrape
3. 해당 데이터를 Google Spreadsheet에 연동하여, 주기적으로 업데이트
4. 2~3의 절차를 매일 운영하기 위하여 Google Cloud Platform 사용  
  a. Cloud Scheduler, PubSub, Cloud Functions  
  b. Compute Engine   
  를 사용.  
  위에서 두 부류를 분류한 이유는 a.는 결국 Compute Engine이 정해진 시간에 On/Off가 되기 위한 것이며, b.가 실질적인 일을 하게 되는 곳이기 때문이다.  

---
각 목차로 이동해서 진행해보입시다...

1. [Scrapy Scrape]("https://hoonst.github.io/articles/2020-03/Scrapy-Scrape" "Scrapy Scrape로 이동")  
2. [Google SpreadSheet Setting]("https://hoonst.github.io/articles/2020-03/Google-SpreadSheet-Integration" "Google SpreadSheet Setting로 이동")  
3. [Google Cloud Platform Setting]("https://hoonst.github.io/articles/2020-03/Google-Cloud-Platform-Setting" "Google Cloud Platform Setting로 이동")
