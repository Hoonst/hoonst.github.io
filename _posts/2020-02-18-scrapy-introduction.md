---
layout: post
title:  "Scrapy"
categories: [scrapy]
---
# Scrapy docs 번역
Scrapy를 공부하고 사용하고자 했는데 이상하게 다른 글들이 잘 읽히지 않아 그냥 원문서를 살펴보던 중, 평소 해보고 싶었던 번역 작업 및 블로그 꾸미기를 시작해보겠습니다.

원문: https://docs.scrapy.org/en/latest/intro/overview.html

번역에 재능이 크게 없어가, 원문의 뉘앙스를 최대한 살려가면서 번역했습니다.

# Scrapy at a glance
(Scrapy, 한 눈에 살펴보기)

`Scrapy`는 웹사이트나 데이터마이닝, 정보처리, 역사적 기록 보관을 할 수 있는 데이터들을 크롤링할 수 있는 어플리케이션 프레임워크다.

`Scrapy`는 본디 웹 스크래이핑을 위해 고안되었지만, API들을 이용한 데이터 추출이나 범용적인 웹 크롤러로도 사용가능하다.

# Walk-through of an example spider
(`Spider` 예시를 살펴봐보자)

`Scrapy`가 무엇을 할 수 있는 지 살펴보기 위해 가장 쉽게 `spider`를 사용하면서 `Scrapy Spider`의 예시를 보여주겠다.

아래의 코드는 `Pagination`(page 기능)을 갖춘 http://quotes.toscrape.com 에서 유명한 명언을 `Scrape`하는 코드이다.

~~~python
import scrapy

class QuotesSpider(scrapy.Spider):

    name = 'quotes'
    start_urls = [
        'http://quotes.toscrape.com/tag/humor/',
    ]
    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.xpath('span/small/text()').get(),
            }

        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
~~~

위 코드를 text file에 넣고 `quotes_spider.py`와 같이 이름 지은 뒤, `runspider` 명령어로 실행시켜보자:

~~~
scrapy runspider quotes_spider.py -o quotes.json
~~~

이것이 끝나면, 당신은 명언들의 리스트가 텍스트와 저자가 함께 담겨있는 `quotes.json` 를 갖게 될 것이다.

~~~json
[{
    "author": "Jane Austen",
    "text": "\u201cThe person, be it gentleman or lady, who has not pleasure in a good novel, must be intolerably stupid.\u201d"
},
{
    "author": "Groucho Marx",
    "text": "\u201cOutside of a dog, a book is man's best friend. Inside of a dog it's too dark to read.\u201d"
},
{
    "author": "Steve Martin",
    "text": "\u201cA day without sunshine is like, you know, night.\u201d"
},
...]
~~~

# What just happened?
(시방 뭔일이 일어난 것이여?)

`scrapy runspider quotes_spider.py` 명령을 실행한다면, `Scrapy`는 정의된 `Spider`를 찾고 `crawler engine`을 통해 실행시켰을 것이다.

`Crawl`은 `start_urls` 속성에 정의된 URLs에 `request`를 보내는 것으로 시작할 것이며,(위의 예시에서는 humor 카테고리의 명언이 담긴 URL) `default callback` 함수인 `parse`를 불러, `response object`를 인자로 전달했을 것이다.

`parse` `call back`에서는 `css Selector`를 사용해 명언 요소(element)들을 순회할 것이며, 추출된 명언 텍스트와 저자를 `Python dictionary`에 담아 `yield` 할 것이다. 그 후, 다음 페이지에 대한 링크를 찾을 것이며, 같은 `parse` 함수를 callback 함수로 지정해 새로운 `request`를 만들것이다.

`Scrapy`의 주요 장점에 대해서 하나 더 알려주자면, `request`는 `scheduled` 되고 `processed asynchronously`하게 처리될 것이다. 즉, 하나의 `request`가 처리가 다 되기까지 기다릴 필요가 없고, 하나가 처리될 때 다른 `request`를 전달할 수 있는 것이다. 또한 이 뜻은, 한 `request`가 실패하거나 `error`가 발생하더라도 다른 `request`들은 건재할 것임을 뜻한다.

이런 강점은 빠른 크롤링을 가능하게 하며(동시적인 `requests`를 한번에, 오류 없이 보내는 방법) 몇 가지 `setting`을 통해 크롤링의 우아함에 대한 주도권을 가질 수 있을 것이다.

* 매 `request`간의 `Download Delay`
* Domain, IP별 동시적 `requests` 양 제한
* auto-throttling extension 사용

!Note
위의 예시는 `Feed exports`를 통해 `JSON` file을 사용했는데, 단순히 `XML`이나 `CSV`와 같은 다른 `Export Format`로 변경할 수 있으며, `Storage backend`도 `FTP`나 `Amazon S3`로 변경할 수 있다.
`Item pipeline`으로 데이터 베이스에 `items`를 저장할 수도 있다.

#What else?
(또 뭐?)

`Scrapy`를 통해 웹사이트에서 어떻게 추출하고 items를 저장하는지 살펴보았지만, 이거는 겉만 살짝 핥은 것이다. `Scrapy`는 스크래이핑을 쉽고 효율적으로 만들 수 있는 강력한 기능들이 존재한다.

* `HTML/XML` 소스들에서, `CSS` 선택자나 `XPath` 표현을 통해 데이터를 선택하고 추출하는 빌트인이 지원되며 정규 표현식도 가능.
* `Interactive shell console`(IPython aware): `CSS`와 `XPath` 표현식을 테스트해볼 수 있고, 이는 spider의 디버깅에 큰 효용.
* `Generating Feed Exports`:
다양한 포맷(JSON, CSV, XML)으로 `export`하고 다양한 백엔드(FTP, S3, Local)에 저장 가능.
* 강력한 인코딩 지원과 `auto-detection`을 통해 외국의, 기준 외, 그리고 꺠진 인코딩을 해결할 수 있다.
* `Strong extensibility support`:
`signals`과 `API`(middlewares, extensions and pipelines)를 사용해 다양한 자체 기능을 제작.
* `Built-in extensions and middlewares`:
  * `cookies`와 `session` 관리
  * `Compression`, `Authentication`, `Caching`과 같은 `HTTP Features`
  * `user-agent spoofing`
  * `robots.txt`
  * `crawl depth restriction`
  * 그 이상
* `Telnet console`:
`Scrapy` 프로세스 내에서 작동하고 있는 `Python` 콘솔에 들어감으로써 크롤러를 디버깅하거나 검사
* `Sitemaps`와 `XML/CSV feeds`로부터 재활용이 가능한 `spiders`를 만나볼 수 있으며, `media pipeline`을 통해 `automatically downloading images` 가 가능하고, `DNS resolver`를 캐싱할 수 있습니다. 그리고 더 많아요~

# 다음은 뭔가?
(What's next?)

다음 절차는 `Scrapy` 설치로서 전체 튜토리얼을 따라하여, 완전한 `Scrapy project`를 어떻게 수립하는지 배우고 커뮤니티에도 참여하세요!

관심 가져주셔서 감사합니다.
