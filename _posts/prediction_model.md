# 예측모델 과제 #3

고려대학교 산업경영공학부과 석사과정 2020021326 윤훈상

## [1] 구글 트렌드를 이용하여 Seasonal variations이 있는 데이터 찾기 
https://www.google.com/trends/?hl=ko

Seasonal Variations가 존재하는 데이터를 구글 트렌드에서 조사한 결과, iphone이 해당 특성을 갖고 있는 것으로 나타났다.

 <img src="/Users/yoonhoonsang/Downloads/iphone_search.png" alt="iphone_search" style="zoom: 50%;" />

Iphone 검색어에서 Seasonal Variation이 나타나는 이유는 매년 9월에 새로운 아이폰이 출시가 되기 때문이다. 따라서 한국어로 '아이폰'이라고 검색하여 국내의 검색 패턴을 비교해보아도 비슷한 양상을 띄는 것을 볼 수 있다. Iphone의 Seasonal Variation은 incremental한 것이 아닌 Constant한 것으로 나타난다. 

원래 계획은 구글 트렌드에서 데이터를 제공하는 범위인 2004년 부터 데이터를 가져오려고 했으나, 

1) '아이폰' 검색어와의 비교 (아이폰이 한국에서는 2011년 정도에 보편화가 시작)

2) 2011

또한, 모델을 구현하여 Predict를 할 때, 2020년 현재의 데이터에 대한 예측을 진행하려 했으나, 

1) 현재 2020년이 10월까지 진행이 안 된 점

2) COVID-19로 인해, 2020년에는 아이폰이 예정되로 9월에 출시되지 않고, 연기 발표를 하는 것과 같은 특이 현상이 나타나 훈련은 2018년까지 진행하고 2019년을 Test Data로 활용하고자 한다.

![image-20201004214735512](/Users/yoonhoonsang/Library/Application Support/typora-user-images/image-20201004214735512.png)

 (1)  다음 방법들을 적용하고 예측력 비교해 보세요 (최소 10시점 이후 예측할 것): 

## Dummy variable (Binary Variable Models)

```python
iphone_dummy = iphone_train.copy()

# month to one hot encoding
iphone_dummy['month'] = pd.DatetimeIndex(iphone_dummy['date']).month
iphone_dummy = pd.get_dummies(iphone_dummy, columns = ['month'])

# make time by index, but drop December
iphone_dummy['time'] = iphone_dummy.index + 1
iphone_dummy = iphone_dummy.drop(['month_12'], axis = 1)

X_train = iphone_dummy.drop(['count', 'date'], axis =1)
y_train = iphone_dummy['count']

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train)
fitted = model.fit()

iphone_dummy_test = iphone_test.copy()

iphone_dummy_test['month'] = pd.DatetimeIndex(iphone_dummy_test['date']).month
iphone_dummy_test = pd.get_dummies(iphone_dummy_test, columns = ['month'])

iphone_dummy_test['time'] = iphone_dummy_test.index + 1
iphone_dummy_test = iphone_dummy_test.drop(['month_12'], axis = 1)

X_test = iphone_dummy_test.drop(['count', 'date'], axis =1)
y_test = iphone_dummy_test['count']

X_test = sm.add_constant(X_test)
y_predict = fitted.predict(X_test)
mean_squared_error(y_test, y_predict)
# MSE = 139.81684027778203


fitted.summary()
```

<img src="/Users/yoonhoonsang/Library/Application Support/typora-user-images/image-20201004221618323.png" alt="image-20201004221618323" style="zoom:50%;" />

## Trigonometric Models

### Trigonometric #1 (sine, cosine term 1개씩 있는 방법)

```python
def simple_exponential_smoothing(df, alpha):
    initial = np.mean(np.array(df['count']))

    levels = [initial]

    counts = df['count']

    for count in counts:
        level = alpha * (count) + (1-alpha) * levels[-1]
        levels.append(level)
        
    return levels[-1]

last_level = simple_exponential_smoothing(iphone_exp1, 0.2)

y_test = iphone_test['count']
y_predict = [last_level] * len(y_test)

mean_squared_error(y_test, y_predict)
```



### Trigonometric #2 (sine, cosine term 2개씩 있는 방법)

```python
def trigonometric_2(df):
    def preprocess(df):
        pi = math.pi
        L = 12

        df['time'] = df.index + 1
        df['sin1'] = df.apply(lambda row: math.sin((2*pi*(row.time)) / L), axis = 1)
        df['cos1'] = df.apply(lambda row: math.cos((2*pi*(row.time)) / L), axis = 1)
        
        df['sin2'] = df.apply(lambda row: math.sin((4*pi*(row.time)) / L), axis = 1)
        df['cos2'] = df.apply(lambda row: math.sin((4*pi*(row.time)) / L), axis = 1)
        
        X = df.drop(['count', 'date'], axis =1)
        y = df['count']
        
        return X, y
    
    X_train, y_train = preprocess(df)
    
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    fitted = model.fit()    
    return fitted
```



## 구간 평균법

```python
N = 3


def moving_average(df):
    means = []
    
    count = df['count']
    for index in df.iloc[N-1:].index:
        count_mean = np.mean(count[index-N+1:index+1])
        means.append(count_mean)
        
    return means[-1]

last_average = moving_average(iphone_ma)

y_test = iphone_test['count']
y_predict = [last_average] * len(y_test)

mean_squared_error(y_test, y_predict)
```



## Simple exponential smoothing

|      |      |      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |      |



## Double exponential smoothing 

## Additive Holt-Winters

## Multiplicative Holt-Winters

## (2) Additive Holt-Winters와 Multiplicative Holt-Winters 방법에서 Weighting parameter (alpha, gamma, delta) 변경하여 예측력 비교해 보세요.





[2] 구글 트렌드를 이용하여 Trend가 존재하는 데이터 찾기 (Seasonal Variations은 없는 것) [https://www.google.co](https://www.google.com/trends/?hl=ko)

 (1) 구간평균법, Simple exponential smoothing과 Double exponential smoothing 방법 적용하고 예측력 비교해 보세요 (최소 10시점 예측)



 (2)  Simple exponential smoothing 와 Double exponential smoothing 방법에서 Weighting parameter를 변경하여 예측력이 어떻게 변하는지 살펴 보시오. 



