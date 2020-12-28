---
layout: post
title:  "Scrapy Selector"
categories: [scrapy]
---
# Selector

웹페이지를 스크래이핑할 때, 가장 많이 마주해야 하는 일은 HTML 소스에서 데이터를 긁어오는 것이다. 이때 사용가능한 여러 라이브러리가 존재하는데, 예를 들면,

* BeautifulSoup:
Python 프로그래머들에게 인기있는 웹 스크레이핑 라이브러리로서, HTML 코드 구조,  Python object를 형성하며, 나쁜 `Markup language`를 잘 처리하지만 단점이 하나 있다(느리다).
  * (역자 주: `Markup`과 `Markdown` 뭐가 다른가?...)

* lxml은 XML 파싱 라이브러리(HTML도 가능)로서 Pythonic API 인 ElementTree에 기반한다.

Scrapy는 데이터를 추출함에 있어 고유의 메커니즘을 활용한다. 이는 XPath 또는 CSS 표현에 의하여 HTML의 특정 부분을 선택하기 때문에 `selectors` 라고 불린다.

`XPath`는 XML 문서의 노드를 선택하는 언어이며, HTML에서도 활용 가능하다. CSS는 본디 HTML 문서의 스타일을 입히는 언어이다. 이를 활용해 `selectors`로 하여금 스타일과 HTML elements를 연루시키도록 한다.

~~~
!Note
`Scrapy selectors`는 `Parsel` 라이브러리의 포장지다; 이 포장지의 목적은 `Scrapy Response Objects`대한 융합을 원활하게 하기 위해서이다.

`parsel`은 `Scrapy`와는 단독으로 활용가능한 웹 스크래이핑 라이브러리다. `lxml`을 내부에서 활용하며, `lxml API` 위에 쉬운 API를 얹는다. 즉, `Scrapy Selectors`는 lxml과 비슷한 속도와 파싱 정확도를 갖는다.
~~~

# Using selectors
(선택자 활용하기)

## Constructing Selectors
(선택자 구성하기)

~~~
>>> response.selector.xpath('//span/text()').get()
'good'
~~~
> Response의 selector의 xpath를 통해 get한다

XPath와 CSS로 Response에 대하여 쿼리 하는 경우가 매우 흔하기 때문에 두 개의 단축어가 있다.
`response.xpath()` and `response.css`

~~~
>>> response.xpath('//span/text()').get()
'good'
>>> response.css('span::text').get()
'good'
~~~

Scrapy selectors는 `Selector` 클래스의 인스턴스로서 `TextResponse` object 또는 markup를 넘겨줌으로써 unicode string으로 구성된다. 주로 직접 `Scrapy selectors`를 구성할 필요가 없다: `response` 오브젝트는 Spider callbacks에서 사용가능하여 많은 경우 `response.css()` 또는 `response.xpath()`를 활용하는 것이 편하다. `response selector`나 단축어중의 하나를 활용하는 것으로 `response body`가 딱 한번 파싱이 되도록 할 수 있다.

만일 필요하다면, `Selector`를 직접 활용하는 것이 가능하다.

* 텍스트에서 구성하기

~~~python
>>> from scrapy.selector import Selector
>>> body = '<html><body><span>good</span></body></html>'
>>> Selector(text=body).xpath('//span/text()').get()
'good'
~~~
* response에서 구성하기
-`HtmlResponse`는 `TextResponse`의 하위 클래스이다.

~~~python
>>> from scrapy.selector import Selector
>>> from scrapy.http import HtmlResponse
>>> response = HtmlResponse(url='http://example.com', body=body)
>>> Selector(response=response).xpath('//span/text()').get()
'good'
~~~

`Selector`는 자동으로 XML과 HTML중, input type에 맞춰 가장 좋은 파싱 규칙을 고른다.

# Using selectors
selectors에 대하여 설명하기 위해 `Scrapy shell`(Interactive testing 제공) 와 Scrapy 설명서 서버에 있는 예시 페이지를 활용하겠다.

https://docs.scrapy.org/en/latest/_static/selectors-sample1.html

완벽함을 위해(열지 못하는 경우), HTML 코드를 아래에 표시한다.

~~~HTML
<html>
 <head>
  <base href='http://example.com/' />
  <title>Example website</title>
 </head>
 <body>
  <div id='images'>
   <a href='image1.html'>Name: My image 1 <br /><img src='image1_thumb.jpg' /></a>
   <a href='image2.html'>Name: My image 2 <br /><img src='image2_thumb.jpg' /></a>
   <a href='image3.html'>Name: My image 3 <br /><img src='image3_thumb.jpg' /></a>
   <a href='image4.html'>Name: My image 4 <br /><img src='image4_thumb.jpg' /></a>
   <a href='image5.html'>Name: My image 5 <br /><img src='image5_thumb.jpg' /></a>
  </div>
 </body>
</html>
~~~

그럼 Shell을 열어보자!

~~~
scrapy shell https://docs.scrapy.org/en/latest/_static/selectors-sample1.html
~~~

Shell이 켜진다면, `response` 쉘 변수에 response가 담겨있을 것이고, 그에 해당하는 selector가 `response selector`에 담겨있을 것이다.

우리가 HTML을 다르고 있기 떄문에, selector는 자동적으로 HTML parser를 사용할 것이다.

따라서, 페이지의 HTML code를 살펴보며, title tag 안에 있는 text를 선택하기 위한 XPath를 구성해보자.

~~~python
>>> response.xpath('//title/text()')
[<Selector xpath='//title/text()' data='Example website'>]
~~~

텍스트 데이터를 추출하기 위해선 `.get()` 또는 `.getall()` 함수를 활용한다.

~~~python
>>> response.xpath('//title/text()').getall()
['Example website']
>>> response.xpath('//title/text()').get()
'Example website'
~~~

`.get()`은 항상 단일 결과를 내놓으며, 만약 복수라면 가장 첫 번째 일치를 보여준다.;
만약 match가 존재하지 않는다면, None이 나타난다.
`.getall()`은 리스트로서 결과를 내놓는다.

CSS selectors는 텍스트나 속성 노드를 CSS3 pseudo-elements를 통해 선택 가능하다.

~~~python
>>> response.css('title::text').get()
'Example website'
~~~

보다시피, `.xpath()`와 `.css()` 함수는 `SelectorList` 인스턴스를 리턴하고, 이는 새로운 선택자들의 리스트이다. 이 API는 nested data에서 선택할 때 활용된다.

~~~python
>>> response.css('img').xpath('@src').getall()
['image1_thumb.jpg',
 'image2_thumb.jpg',
 'image3_thumb.jpg',
 'image4_thumb.jpg',
 'image5_thumb.jpg']
 ~~~

 만약 첫 일치 element만을 추출하고 싶다면, `.get()` selector를 부르도록하자. (별칭 `.extract_first()`는 Scrapy의 이전 버전에서 활용되었다.)

~~~python
>>> response.xpath('//div[@id="images"]/a/text()').get()
'Name: My image 1 '
~~~

만약 `None`이 리턴된다면 조건을 만족하는 element가 존재하지 않는 것이다.

~~~python
>>> response.xpath('//div[@id="not-exists"]/text()').get() is None
True
~~~

디폴트 값을 미리 전달하여, `None`이 나타나는 것을 대신할 수 있다.

~~~python
>>> response.xpath('//div[@id="not-exists"]/text()').get(default='not-found')
'not-found'
~~~

예를 들어,`@src`를 사용하는 것 대신에, XPath는 Selector의 `.attrib` 속성을 사용하여 원하는 속성을 쿼리를 통해 찾을 수 있다.

~~~python
>>> [img.attrib['src'] for img in response.css('img')]
['image1_thumb.jpg',
 'image2_thumb.jpg',
 'image3_thumb.jpg',
 'image4_thumb.jpg',
 'image5_thumb.jpg']
 ~~~

 단축어로서, `.attrib`는 `SelectorList`에 직접적으로 활용될 수 있으며, 가장 첫 속성을 return 한다.

~~~python
>>> response.css('img').attrib['src']
'image1_thumb.jpg'
~~~

이는 id로 선택하거나, 웹페이지의 단일 element를 선택할 때와 같이, 단일의 결과가 예상될 때 효용성이 크다.

~~~python
>>> response.css('base').attrib['href']
'http://example.com/'
~~~

이제 우리는 base url과 몇 이미지 링크를 가져와볼 것이다.

~~~python
'http://example.com/'

>>> response.css('base::attr(href)').get()
'http://example.com/'

>>> response.css('base').attrib['href']
'http://example.com/'

>>> response.xpath('//a[contains(@href, "image")]/@href').getall()
['image1.html',
 'image2.html',
 'image3.html',
 'image4.html',
 'image5.html']

>>> response.css('a[href*=image]::attr(href)').getall()
['image1.html',
 'image2.html',
 'image3.html',
 'image4.html',
 'image5.html']

>>> response.xpath('//a[contains(@href, "image")]/img/@src').getall()
['image1_thumb.jpg',
 'image2_thumb.jpg',
 'image3_thumb.jpg',
 'image4_thumb.jpg',
 'image5_thumb.jpg']

>>> response.css('a[href*=image] img::attr(src)').getall()
['image1_thumb.jpg',
 'image2_thumb.jpg',
 'image3_thumb.jpg',
 'image4_thumb.jpg',
 'image5_thumb.jpg']
 ~~~
