---
layout: post
title:  "Selecting dynamically-loaded content
"
categories: [scrapy]
---
출처: https://docs.scrapy.org/en/latest/topics/dynamic-content.html

최근에 맡은 프로젝트 중에서 인터넷 강사들의 수강후기와 QnA 갯수를 크롤링하여 수치 변화 파악 및 강사의 인기도를 측정해보는 프로젝트를 진행하고있다.

평소에는 Selenium 만을 사용하여 크롤링을 사용하였지만 아무래도 Selenium이 무겁고, 뭔가 배포를 위해서는 드라이버까지 따라 들어와야 하는 어려움이 있기 때문에, 이 참에 크롤링 계의 끝판왕인 프레임워크 Scrapy를 본격 공부해보기로 했다. 사실 본 번역 시리즈의 이유도 여기에 있다.

처음에 본 프로젝트를 진행함에 있어 `Simple is the best`라는 개발의 신조를 적용하기 위하여, Requests와 BeautifulSoup로 가능 여부를 파악하려 했다. 하지만 모든 인강 사이트가 동적으로 로딩이 이루어지는 사이트라 response에 table이 담겨있지 않았다. 다운로드 시간때문에 그런가 싶어서 wait를 걸어줬지만 이 역시 먹히지 않았다.

따라서 Selenium으로 진행해보았다.
웹 드라이버는 requests와 다르게 동적으로 렌더링 되는 것도 모두 response에 담기기 때문에 작동은 했지만 본 프로젝트는 웹드라이버의 흔적을 제거한채 진행하고 싶어서 본 문서를 번역해보고자 한다.

# Selecting dynamically-loaded content

몇 웹페이지들은 브라우저를 로드할 때 나타난다. 하지만, `Scrapy`로 이를 다운로드 할 때, `Selector`로 이를 접근할 수 없는 경우가 발생한다.

이 때, 추천하는 방식은 *Data source를 찾고* 거기서 데이터를 추출하는 것이다.

만약 그것에 실패한다면, 웹브라우저의 DOM을 통해 접근하는 것이다. 아래의 Pre-rendering JavaScript 섹션을 살펴보라.

# Finding the data source
(데이터 소스를 찾아라!)

원하는 데이터를 추출하기 위해선, 소스의 위치를 알아야 한다.

만약 non-text-based 포맷의 데이터라면, 이미지나 PDF 같은, 웹브라우저의 network tool을 활용해 일치하는 request를 찾고 재생산하라.

만약 웹브라우저가 원하는 데이터를 text로 보여주고 있다면 데이터는 embedded JavaScript 코드로 이루어져 있거나, 텍스트 기반 포맷의 외부 리소스에서 로드 되고 있을 것이다.

그 경우에는 wgrep과 같은 툴을 활용해 리소스의 URL을 활용할 수 있다.

만약 데이터가 기존 URL에서 나온다면, 웹페이지의 소스코드를 살펴봄을 통해 데이터의 위치를 파악해야 한다.

만약 데이터가 다른 URL에서 온다면, Request를 다시 만들어야 한다.

# Inspecting the source code of a webpage

가끔 DOM이 아닌 웹페이지의 소스코드를 검사하여 데이터의 위치를 파악해야 할 때가 있다.

`Scrapy`의 `fetch` 명령어를 통해 웹페이지 내용을 다운로드 한다.

~~~
scrapy fetch --nolog https://example.com > response.html
~~~


if: 데이터가 자바스크립트 내 `<script/>` element 내에 있다면 Parsing JavaScript code 섹션을 봐야한다.

elif: 원하는 데이터를 찾을 수 없다면, Scrapy 뿐아니라 curl 또는 wget과 같은 HTTP client를 통해 웹페이지를 다운로드 해보고 정보가 response에 존재하는지 찾아보자.

elif: 다른 client에서는 데이터가 존재한다면 Scrapy의 `request`를 해당 HTTP client로 전환해야 한다. 예를 들어 같은 user-agent string(User-Agent) 또는 같은 `headers`를 활용해본다.

else: Reproducing Requests 섹션

# Reproducing Requests

가끔 우리는 웹브라우저가 하는 방식대로 Request를 다시 제작해야 하는 경우가 존재한다.

웹브라우저의 Network tool을 사용하여 어떻게 request를 던지는지 살펴보고 Scrapy에서 같은 방식으로 던져보라.

HTTP method와 URL이 같은 `Request`를 산출해낼 수도 있다. 하지만 아마 새로운 body, headers 그리고 Form parameters(`FormRequest` 섹션)를 바꿔야 할 것이다.

원하는 response를 얻었다면 원하는 데이터를 가져올 수 있을 것이다.

Scrapy를 통해 어떤 request든 재생산이 가능할 것이다. 하지만 가끔 모든 request를 재생산하는 것은 비효율적으로 보일 수도 있다. 만약 해당 경우이며, 크롤링 스피드가 큰 고려사항이 아니라면 JavaScript pre-rendering이 다른 대안이 될 것이다.

만약 원하는 response가 가끔만 등장한다면, 당신의 request의 문제라기보다 서버의 문제일 것이다. 타겟 서버가 버그가 있거나, 오버로드되거나 또는 너의 request를 튕겨낼 수도 있다.

# Handling different response formats

원하는 데이터의 response를 얻었다면 response 타입에 따라 원하는 데이터를 가져오는 방식이 다를 것이다.

Response Type

* HTML / XML: selectors 사용
* JSON: response.text로부터 json.loads
~~~
data = json.loads(response.text)
~~~
if) 원하는 데이터가 JSON으로 HTML / XML 코드에 포함이 되어 있다면, Selector를 통해 부를 수 있다.
~~~
selector = Selector(data['html'])
~~~

* JavaScript 또는 `<script/>`에 원하는 데이터가 있는 경우: Parsing JavaScript Code
* CSS: `response.text` 에 정규표현식 적용
* Image or PDF: Response.text로부터 Response를 바이트로 읽은 후, OCR 솔루션으로 텍스트에서 데이터를 가져온다.

예시: pytesseract / tabula-py

* SVG: selectors로 충분히 가능한 것이, SVG또한 XML 기반이기에.
또는 raster image로 SVG 코드를 변환한 후에 그 이미지를 처리하는 것이 좋을 수 있다.

# Parsing JavaScript code

원하는 정보가 JavaScript에 하드코딩되어 있을 경우, JavaScript 코드를 가져오는 것이 최우선이다.

* JavaScript 코드가 JavaScript file로 되어 있는 경우, response.text를 읽는다.
* Javascript 코드가 HTML 내의 `<script/>` element 안에 있다면 selectors를 통해 text를 가져온다.

JavaScript 코드를 통해 string을 가져왔다면, 원하는 데이터를 가져올 수 있을 것이다.

* JSON 포맷의 데이터는 정규표현식을 고려해볼 수도 있다. 그 후, json.loads로 파싱을 할 수 있다.

예를 들어 만약 JavaScript 코드가

`var data = {"field": "value"};` 를 포함하고 있을 때, 이는 다음과 같이 뽑아 낼 수 있다.

~~~python
>>> pattern = r'\bvar\s+data\s*=\s*(\{.*?\})\s*;\s*\n'
>>> json_data = response.css('script::text').re_first(pattern)
>>> json.loads(json_data)
{'field': 'value'}
~~~

* 다른 경우, js2xml를 사용하여 JavaScript 코드를 XML document로 변환한 뒤 selectors로 파싱하라.

예를 들어, `var data = {"field": "value"};`가 JavaScript에 포함되어 있다면, 다음과 같이 데이터를 뽑아낼 수 있다.

~~~python
>>> import js2xml
>>> import lxml.etree
>>> from parsel import Selector
>>> javascript = response.css('script::text').get()
>>> xml = lxml.etree.tostring(js2xml.parse(javascript), encoding='unicode')
>>> selector = Selector(text=xml)
>>> selector.css('var[name="data"]').get()
'<var name="data"><object><property name="field"><string>value</string></property></object></var>'
~~~

# Pre-rendering Javascript

추가적인 requests를 진행하는 웹페이지에서는 그 request들을 재생산해내는 것이 추천하는 방식이다. 노력한 결과는 구조적이고, 파싱이 적게 필요한 완벽한 데이터를 얻는 효용성이 있을 것이다.

하지만 가끔 이런 requests를 재생산하는게 어려울 수 있다. 또는 어떤 request도 당신이 원하는 데이터를 얻게 해줄 수 없을 수 있다. 웹브라우저의 스크린샷과 같이 말이다.

이때는 Splash JavaScript-rendering 서비스를 사용하고, scrapy-splash를 통해 깔끔한 통합을 이룩하라.

Splash는 webpage의 DOM을 HTML 형식으로 내놓음으로서 selector를 통해 파싱이 가능하다. Splash는 스크립트와 환경설정을 통해 유연함을 갖고 있다.

만일, Splash가 제공하는 것 이상을 원한다면, 예를 들어 Python을 통해 DOM과의 상호작용을 하고 싶거나, 다중의 웹브라우저를 띄우고 싶다면 Headless browser 사용을 권장한다.

# Using a headless browser

Headless browser는 자동화를 위한 특별한 웹브라우저이다.

가장 쉬운 방법은 Selenium을 활용하는 것이며, scrapy-selenium으로 완벽하게 구상하기를 바란다.

이상전달끝
 
