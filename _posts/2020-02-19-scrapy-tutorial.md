---
layout: post
title:  "Scrapy Tutorial"
categories: [scrapy]
---
*출처: https://docs.scrapy.org/en/latest/intro/tutorial.html

#Scrapy Tutorial

본 튜토리얼에선, 설치가 완료되어 있다고 생각하고 진행하겠다. 아니면 설치 가이드를 살펴보길 바란다.

우리는 유명한 작가들의 인용구들의 리스트가 있는 웹사이트,
[quotes.toscrape.com](http://quotes.toscrape.com/)
를 스크래이프 하고자한다.

이 튜토리얼은 일련의 과정을 거칠 것이다 :

1. 새로운 Scrapy 프로젝트 생성
2. 사이트를 탐색하고 데이터를 추출하는 `spider` 생성
3. 스크래이프한 데이터를 커맨드라인으로 추출
4. 이따르는 링크들에 대하여 재귀적으로 `spider`를 전환
5. `spider` 인자를 사용

`Scrapy`는 파이썬으로 작성되어 있다. 만일 해당 언어에 대한 뉴비라면, 어떤 언어인지 먼저 파악하고 오는 것이 `Scrapy`를 배우는 것에 더 큰 도움이 될 수 있다.

만일 다른 언어에는 익숙하고 `Python`을 빠르게 익히고 싶다면, [Python Tutorial](https://docs.python.org/3/tutorial)이 좋은 자료가 될 것이다.

그리고 완전 프로그래밍 자체에 대한 뉴비이고, 파이썬으로 시작해보고 싶다면 다음의 책들이 유영할 것이다.

* [Automate the Boring Stuff With Python](https://automatetheboringstuff.com/)
* [How To Think Like a Computer Scientist](http://openbookproject.net/thinkcs/python/english3e/)
* [Learn Python 3 The Hard Way](https://learnpythonthehardway.org/python3/)

또한, [비개발자를 위한 Python 자료](https://wiki.python.org/moin/BeginnersGuide/NonProgrammers)나 [learnpython-subreddit에서 추천하는 자료들](https://www.reddit.com/r/learnpython/wiki/index#wiki_new_to_python.3F)도 좋을 수 있다.

# Creating a project
(프로젝트 시작!)

스크래이핑을 시작하기 전, `Scrapy` 프로젝트를 셋팅해야한다. 코드를 저장할 디렉토리에 들어간 후 아래를 실행하라:

~~~
scrapy startproject tutorial
~~~

이는 `tutorial` 디렉토리와 다음의 내용들이 포함될 것이다.

~~~
tutorial/
    scrapy.cfg            # deploy configuration file

    tutorial/             # project's Python module, you'll import your code from here
        __init__.py

        items.py          # project items definition file

        middlewares.py    # project middlewares file

        pipelines.py      # project pipelines file

        settings.py       # project settings file

        spiders/          # a directory where you'll later put your spiders
            __init__.py
~~~

# Our first Spider
(첫 스파이더맨)

`Spiders`는 정의하는 클래스이며 `Scrapy`가 이를 활용해 웹사이트 (또는 웹사이트 묶음)의 정보들은 스크래이핑 한다. 이는 `Spider`를 상속받아야 하며, 최초 Requests를 정의해야 하고, 페이지 링크들을 어떻게 이어서 따라다니고, `parse`의 방법, 그리고 추출방법을 고려해야 한다.

아래의 코드는 우리의 첫 `Spider`다. `tutorial/spiders` 디렉토리리 아래의 `quotes_spider.py`에 저장한다.

~~~python
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
~~~

보다시피, 우리의 `Spider`는 `scrapy.Spider`를 상속받고 몇가지 속성과 함수를 정의한다.

* `name`: `Spider`를 표시. 프로젝트 내에서 중복이 되면 안된다. (역자 주: 실수로 `name`을 중복으로 실행해봤더니 그 어느것도 실행하지 못하는 결과가 나타난다)
* `start_requests()`: `Spider`가 탐색할 수 있는 여러 `Request`를 리턴해야한다 (`Request`의 리스트를 리턴할 수 도 있지만 제너레이터 함수를 활용할 수도 있다). 이후의 `requests`는 최초의 `requests`이후에 연속적으로 생성될 것이다.
* `parse()`: `requests`로 다운 받은 `response` 들을 처리할 함수이다. `response` 파라미터는 페이지의 내용이나 다른 기능들을 갖고 있는 `TextResponse`의 인자이다.

`parse()`는 `response`를 파싱하며, 데이터를 `dictionary` 형태로 추출하고 새로운 `requests`를 보내기 위한 새로운 `URL`들을 탐색하기도 한다.

# How to run our spider
(어떻게 스파이더맨을 돌릴까?)

`Spider`가 작동하게 하려면, 프로젝트 폴더의 최상단 디렉토리에서 아래의 코드를 실행하라:
~~~
scrapy crawl quotes
~~~
이 커맨드는 `quotes`의 이름을 가진 `spider`를 실행하, `quotes.toscrape.com`에 `requests`를 보낼 것이다. 그리고 아래와 유사한 결과를 얻을 것이다.

~~~
... (omitted for brevity)
2016-12-16 21:24:05 [scrapy.core.engine] INFO: Spider opened
2016-12-16 21:24:05 [scrapy.extensions.logstats] INFO: Crawled 0 pages (at 0 pages/min), scraped 0 items (at 0 items/min)
2016-12-16 21:24:05 [scrapy.extensions.telnet] DEBUG: Telnet console listening on 127.0.0.1:6023
2016-12-16 21:24:05 [scrapy.core.engine] DEBUG: Crawled (404) <GET http://quotes.toscrape.com/robots.txt> (referer: None)
2016-12-16 21:24:05 [scrapy.core.engine] DEBUG: Crawled (200) <GET http://quotes.toscrape.com/page/1/> (referer: None)
2016-12-16 21:24:05 [scrapy.core.engine] DEBUG: Crawled (200) <GET http://quotes.toscrape.com/page/2/> (referer: None)
2016-12-16 21:24:05 [quotes] DEBUG: Saved file quotes-1.html
2016-12-16 21:24:05 [quotes] DEBUG: Saved file quotes-2.html
2016-12-16 21:24:05 [scrapy.core.engine] INFO: Closing spider (finished)
...
~~~

그리고 폴더를 보면 `quotes-1.html`와 `quotes-2.html`의 새로운 파일들이 존재하는 것일 볼 수 있으며, `parse`가 다루게 될, 각각의 URL에 대한 컨텐츠를 얻을 수 있을 것이다.

# What just happened under the hood?
(아니 그래서 지금 뭔일이 일어났당가?)

`Scrapy`는 `start_requests`가 return한 `scrapy.Request`를 스케줄링 할 것이다. 각각의 `Response`를 인스턴스화 하고 각각의 `request`와 연루된 콜백 함수를 불러, `Response`를 인자로서 보내버린다.

# A shortcut to the start_requests method
`start_requests()`함수를 통해 `URL`의 `scrapy.Request` 오브젝트를 생성하기보다, `URL`리스트로 구성된 `start_urls` 클래스 속성을 정의할 수 있다.

~~~python
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
        'http://quotes.toscrape.com/page/2/',
    ]

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
~~~

`parse()` 함수는 `Scrapy`가 명시적으로 언급하지 않아도 나타나, 각 URL에 대한 `requests`들을 처리하기 위해서 불려질 것이다. 이는 `parse()` 가 `Scrapy`의 디폴트 콜백 함수이기 떄문이다.

# Extracting data

`Scrapy`로 데이터를 추출하는 방법을 가장 잘 습득하는 법은 `Scrapy Shell`을 통해 `selectors`를 사용해보는 것이다.

~~~
scrapy shell 'http://quotes.toscrape.com/page/1/'
~~~

~~~
!Note
Scrapy shell을 사용할 때, 언제나 url의 따옴표를 닫는 것을 주의하라!. 안그러면 & 문자를 통해 전달되는 argument를 포함하는 url을 이해 못할 수도 있다.
~~~

그럼 결과로 아래와 같은 메시지 등장!

~~~
[ ... Scrapy log here ... ]
2016-09-19 12:09:27 [scrapy.core.engine] DEBUG: Crawled (200) <GET http://quotes.toscrape.com/page/1/> (referer: None)
[s] Available Scrapy objects:
[s]   scrapy     scrapy module (contains scrapy.Request, scrapy.Selector, etc)
[s]   crawler    <scrapy.crawler.Crawler object at 0x7fa91d888c90>
[s]   item       {}
[s]   request    <GET http://quotes.toscrape.com/page/1/>
[s]   response   <200 http://quotes.toscrape.com/page/1/>
[s]   settings   <scrapy.settings.Settings object at 0x7fa91d888c10>
[s]   spider     <DefaultSpider 'default' at 0x7fa91c8af990>
[s] Useful shortcuts:
[s]   shelp()           Shell help (print this help)
[s]   fetch(req_or_url) Fetch request (or URL) and update local objects
[s]   view(response)    View response in a browser
>>>
~~~

Shell을 사용하여, CSS와 `response object`를 통해 요소들을 선택할 수 있다.

~~~
>>> response.css('title')
[<Selector xpath='descendant-or-self::title' data='<title>Quotes to Scrape</title>'>]
~~~ 
