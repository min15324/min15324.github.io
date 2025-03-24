---
layout: single
title: 캐글의 Titanic - Machine Learning from Disaster Competition 참여하기
---

Kaggle의 타이타닉 데이터셋을 활용한 생존자 예측 모델 만들기에 도전해 보고자 한다.

넘파이, 판다스, 사이킷런 등을 활용해야 한다.


```python
import pandas as pd
import numpy as np
import sklearn
import sys

assert sklearn.__version__ >= "1.0.1"
assert sys.version_info >= (3, 7)

np.random.seed(42)
```

구글 드라이브에 저장해둔 타이타닉 데이터셋을 불러온다.


```python
gender_submission = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Files/gender_submission.csv")
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Files/test.csv")
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Files/train.csv")
```

gender_submission은 제출해야 할 파일의 예시이다. PassengerId에 따른 생존 여부가 Survived 열에 저장되어 있다(1은 생존, 0은 사망)


```python
gender_submission.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  418 non-null    int64
     1   Survived     418 non-null    int64
    dtypes: int64(2)
    memory usage: 6.7 KB



```python
gender_submission.head()
```





  <div id="df-034421f3-7007-45e8-a8eb-30dea5c13762" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-034421f3-7007-45e8-a8eb-30dea5c13762')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-034421f3-7007-45e8-a8eb-30dea5c13762 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-034421f3-7007-45e8-a8eb-30dea5c13762');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4a50db8b-e39a-4db4-a5ad-d858814c71fb">
  <button class="colab-df-quickchart" onclick="quickchart('df-4a50db8b-e39a-4db4-a5ad-d858814c71fb')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4a50db8b-e39a-4db4-a5ad-d858814c71fb button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




test는 훈련한 모델의 성능 검증을 위한 테스트 셋이다.


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.1+ KB


train은 훈련에 사용될 훈련 셋이다.


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
train["Ticket"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>373450</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>211536</td>
    </tr>
    <tr>
      <th>887</th>
      <td>112053</td>
    </tr>
    <tr>
      <th>888</th>
      <td>W./C. 6607</td>
    </tr>
    <tr>
      <th>889</th>
      <td>111369</td>
    </tr>
    <tr>
      <th>890</th>
      <td>370376</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 1 columns</p>
</div><br><label><b>dtype:</b> object</label>



생존 유무에 특별한 영향을 주지 않는다고 판단되는 특성을 제거한다.

승선 장소인 Embarked, 승객 번호인 PassengerId, 이름인 Name은 승객을 분류하는 데는 유용하지만 생존 확률을 계산하는 데에는 부적절해 보인다. 다만 PassengerId의 경우 제출할 csv 파일에 필요하므로 따로 저장해 둔다.

Ticket의 구매한 티켓의 번호 정보가 있다. 티켓 번호에 따라 객실 위치나 등급이 다르다면 생존 결과에 영향이 있을 수 있지만 그런 상관관계가 있는지 알아내기 어렵고 티켓마다 번호의 유형이 달라 학습에 부적절하다고 판단된다.

Cabin은 객실 번호의 정보를 가지고 있다.
객실 번호에 따라 객실의 위치도 다를 것이고 객실의 위치에 따라 대피 난이도도 달랐을 것이다. 중요하게 활용할 수 있는 특성이지만 결측치가 너무 많다. 적절한 값으로 대체한다 하더라도 그 양이 너무 커 데이터에 영향을 줄 수 있으므로 특성을 삭제하는 것이 나아 보인다.


```python
PassengerId = test['PassengerId']
PassengerId
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 1 columns</p>
</div><br><label><b>dtype:</b> int64</label>




```python
columns_to_drop = ['Embarked', 'Name', 'Ticket', 'Cabin', 'PassengerId']
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)
```


```python
train
```





  <div id="df-95a2e065-b6cd-420c-8614-8119afbcf9d2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95a2e065-b6cd-420c-8614-8119afbcf9d2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-95a2e065-b6cd-420c-8614-8119afbcf9d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95a2e065-b6cd-420c-8614-8119afbcf9d2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8ba86e9a-7b29-463f-87ae-85c0712e3717">
  <button class="colab-df-quickchart" onclick="quickchart('df-8ba86e9a-7b29-463f-87ae-85c0712e3717')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8ba86e9a-7b29-463f-87ae-85c0712e3717 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_9284f6c7-7a3b-4150-ad06-f11239736d1c">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('train')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_9284f6c7-7a3b-4150-ad06-f11239736d1c button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('train');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
test
```





  <div id="df-d69763ff-2bdf-44bb-a446-6a566820568b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>108.9000</td>
    </tr>
    <tr>
      <th>415</th>
      <td>3</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>416</th>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>417</th>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>22.3583</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d69763ff-2bdf-44bb-a446-6a566820568b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d69763ff-2bdf-44bb-a446-6a566820568b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d69763ff-2bdf-44bb-a446-6a566820568b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-adcf34fb-8239-4933-8925-42b9769262ee">
  <button class="colab-df-quickchart" onclick="quickchart('df-adcf34fb-8239-4933-8925-42b9769262ee')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-adcf34fb-8239-4933-8925-42b9769262ee button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_51d11c5e-3e89-4542-8084-d761be86cf9b">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('test')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_51d11c5e-3e89-4542-8084-d761be86cf9b button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('test');
      }
      })();
    </script>
  </div>

    </div>
  </div>




본격적인 데이터 전처리에 들어간다.
훈련 셋의 경우 Age 특성에, 테스트 셋의 경우 Age와 Fare에 결측치가 존재한다.


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Pclass  418 non-null    int64  
     1   Sex     418 non-null    object 
     2   Age     332 non-null    float64
     3   SibSp   418 non-null    int64  
     4   Parch   418 non-null    int64  
     5   Fare    417 non-null    float64
    dtypes: float64(2), int64(3), object(1)
    memory usage: 19.7+ KB



```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    object 
     3   Age       714 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
    dtypes: float64(2), int64(4), object(1)
    memory usage: 48.9+ KB



```python
train.hist(bins=50, figsize=(12, 8))
```




    array([[<Axes: title={'center': 'Survived'}>,
            <Axes: title={'center': 'Pclass'}>],
           [<Axes: title={'center': 'Age'}>,
            <Axes: title={'center': 'SibSp'}>],
           [<Axes: title={'center': 'Parch'}>,
            <Axes: title={'center': 'Fare'}>]], dtype=object)




    
![png](Untitled1_files/Untitled1_20_1.png)
    


Age 특성의 경우 평균값과 중위수가 별 차이가 없다.
그러나 Fare의 경우 일부 높은 수치가 있어 중위수에 비해 평균값이 크게 높다.

결측치는 모두 중위수로 대체하는 것이 적절해 보인다.


```python
train['Age'].mean()
```




    np.float64(29.69911764705882)




```python
train['Age'].median()
```




    28.0




```python
train['Fare'].mean()
```




    np.float64(32.204207968574636)




```python
train['Fare'].median()
```




    14.4542




```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
```

입력 데이터셋과 타깃 데이터셋을 분리한다.


```python
train_target = train['Survived']
train_data = train.drop('Survived', axis=1)
```

전처리 과정을 파이프라인으로 정의한다.
기본적으로 수치형 특성에는 결측치 처리와 표준화 스케일링만 적용하지만, Parch, SibSp, Fare같이 정규분포를 따르지 않는 특성들은 로그 변환을 추가로 한다.
Sex의 경우 원 핫 인코딩을 통해 수치형 특성으로 변환한다.


```python
num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))
```


```python
preprocessing = ColumnTransformer([
        ("log", log_pipeline, ["Parch", "Fare", "SibSp"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
        ], remainder=num_pipeline)
```

완성된 전처리 과정을 사용해 여러 회귀 모델에 적용해 본다. 사용할 모델은 선형 회귀 모델, 결정 트리 회귀 모델, 랜덤 포레스트 회귀 모델이다.


```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(train_data, train_target)
```




<style>#sk-container-id-7 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-7 {
  color: var(--sklearn-color-text);
}

#sk-container-id-7 pre {
  padding: 0;
}

#sk-container-id-7 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-7 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-7 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-7 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-7 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-7 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-7 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-7 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-7 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-7 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-7 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-7 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-7 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-7 div.sk-label label.sk-toggleable__label,
#sk-container-id-7 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-7 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-7 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-7 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-7 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-7 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-7 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-7 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-7 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-7 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-79" type="checkbox" ><label for="sk-estimator-id-79" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-80" type="checkbox" ><label for="sk-estimator-id-80" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>columntransformer: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                             SimpleImputer(strategy=&#x27;median&#x27;)),
                                            (&#x27;standardscaler&#x27;,
                                             StandardScaler())]),
                  transformers=[(&#x27;log&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                      func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-81" type="checkbox" ><label for="sk-estimator-id-81" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-82" type="checkbox" ><label for="sk-estimator-id-82" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-83" type="checkbox" ><label for="sk-estimator-id-83" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log1p</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;, func=&lt;ufunc &#x27;log1p&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-84" type="checkbox" ><label for="sk-estimator-id-84" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-85" type="checkbox" ><label for="sk-estimator-id-85" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-86" type="checkbox" ><label for="sk-estimator-id-86" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-87" type="checkbox" ><label for="sk-estimator-id-87" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-88" type="checkbox" ><label for="sk-estimator-id-88" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Pclass&#x27;, &#x27;Age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-89" type="checkbox" ><label for="sk-estimator-id-89" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-90" type="checkbox" ><label for="sk-estimator-id-90" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-91" type="checkbox" ><label for="sk-estimator-id-91" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a></div></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div></div></div>




```python
train_data_prediction = lin_reg.predict(train_data)
train_data_prediction
```




    array([ 6.76292621e-02,  8.87675816e-01,  6.28227765e-01,  8.91347949e-01,
            7.08070409e-02,  1.13330049e-01,  3.48327688e-01,  1.27771408e-01,
            5.92728647e-01,  8.34487607e-01,  6.85804148e-01,  8.00572937e-01,
            1.57469836e-01, -3.83664204e-02,  6.97182526e-01,  6.43101080e-01,
            1.18631308e-01,  2.83862052e-01,  5.61039161e-01,  6.12822088e-01,
            2.74382698e-01,  2.49196934e-01,  6.92327704e-01,  4.81081874e-01,
            5.99185038e-01,  4.73653890e-01,  1.06743340e-01,  4.34316636e-01,
            6.16430175e-01,  1.10439483e-01,  4.00451636e-01,  9.79082489e-01,
            6.15739148e-01,  5.50425742e-02,  4.45986734e-01,  3.43858117e-01,
            1.06767408e-01,  1.51692316e-01,  5.92905158e-01,  6.38532879e-01,
            4.80969643e-01,  7.43104480e-01,  1.10439483e-01,  8.68695877e-01,
            6.68427852e-01,  1.11249679e-01,  6.56419480e-02,  6.15739148e-01,
            3.73961557e-02,  6.35648032e-01,  1.03913603e-01,  1.50371667e-01,
            8.27547547e-01,  7.41204290e-01,  2.65143655e-01,  4.81081874e-01,
            8.21109706e-01,  1.03878648e-01,  8.38626023e-01,  5.27403908e-02,
            1.41432526e-01,  9.66965731e-01,  3.48502430e-01,  1.12604100e-01,
            4.69781872e-01,  3.70232540e-02,  7.74889549e-01,  1.63808173e-01,
            4.64383806e-01,  8.72804260e-03,  2.51478243e-01,  5.29931540e-01,
            4.03117461e-01,  7.41100116e-02,  1.75306559e-01,  1.26451068e-01,
            1.10439483e-01,  1.11249679e-01,  4.32568241e-01,  6.24540105e-01,
            1.50620737e-01,  1.12478268e-01,  6.15940763e-01,  4.94092229e-01,
            8.44219785e-01,  4.69900391e-01,  1.15152927e-01,  1.11249679e-01,
            9.17285305e-01,  1.34359757e-01,  1.05472159e-01,  1.56438661e-01,
            3.28275117e-01,  4.56404199e-02, -7.22166913e-02,  1.11249679e-01,
            2.31543221e-01,  5.08820688e-01,  7.52799453e-01,  2.06237944e-01,
            6.16518231e-01,  1.10439483e-01,  5.29612296e-01,  8.54089235e-02,
           -5.85677311e-02,  1.10439483e-01,  6.55639894e-01,  1.09794906e-01,
            5.26642868e-02,  5.91591819e-01,  3.88892793e-01,  6.46630235e-01,
            1.45914797e-01,  5.98069513e-01,  7.06121216e-01,  1.51036616e-01,
           -1.35884185e-01,  2.25470693e-01,  5.66737713e-01,  6.11647360e-01,
            2.85953429e-01,  1.11249679e-01,  2.21524746e-01,  7.63941961e-01,
            3.38954148e-01,  1.44009171e-01,  1.09660401e-01,  1.29373525e-01,
            5.60220873e-01,  7.07032562e-03,  8.15518851e-02,  1.51949607e-01,
            4.59000356e-01,  7.41204290e-01,  3.01194611e-01,  3.19179626e-01,
            9.81238166e-01,  3.73714162e-01,  1.86296556e-01,  5.41304323e-01,
            6.00711580e-01,  6.50404266e-01,  5.95820342e-01,  1.55936629e-01,
            3.36294469e-01,  2.80814284e-01,  1.15684043e-01,  6.30006595e-01,
            2.21516988e-01,  2.02976777e-01,  1.49351803e-01,  9.76958176e-01,
           -4.76321121e-02,  2.01983288e-02,  1.07242224e-01,  3.45577637e-01,
            6.84979320e-01,  9.96946392e-02,  1.14337036e-01, -7.04647129e-02,
            2.09210683e-02,  7.29065431e-01,  1.21349946e-01,  1.77889752e-01,
            1.38578721e-01,  2.17671145e-01,  9.79453422e-01,  4.35175250e-01,
            4.66737948e-01,  1.98416638e-01,  2.87767017e-01,  1.07076269e-01,
            6.85334579e-01,  1.51036616e-01,  3.12657967e-01,  6.61846897e-02,
           -1.38902014e-02,  8.50355552e-01,  2.72307013e-01, -3.88178530e-02,
            4.35614035e-01,  2.90304366e-01,  6.52900178e-02,  3.44297240e-01,
            7.55813378e-01,  4.96852208e-01,  5.71720695e-01,  3.69601945e-01,
           -3.15750241e-02,  6.42193261e-02,  7.66830721e-01,  3.35859729e-01,
            5.94372654e-01,  3.57454317e-01,  8.83420305e-01,  8.79679173e-01,
            1.09660401e-01,  4.28760664e-03,  6.15739148e-01,  8.13050878e-01,
            1.18255788e-01, -7.04647129e-02,  6.77015789e-02,  5.63674630e-03,
            1.69024875e-01,  7.50798111e-01,  4.35214372e-02,  1.59684876e-01,
            6.85069384e-01,  4.05548589e-01,  1.28839528e-01,  7.70806596e-01,
            1.41551536e-01,  2.72307013e-01,  3.57381269e-02,  9.49711147e-01,
            6.22450245e-01,  1.61732307e-01,  9.99421555e-01,  2.63033282e-01,
            1.80579914e-01,  2.89639572e-01, -2.16332736e-02,  1.10439483e-01,
            3.92452771e-01,  1.52242561e-01,  3.26585998e-01,  1.50217816e-01,
            3.41637249e-01,  4.92188546e-01,  9.12356374e-01,  1.04017387e-01,
            1.06413292e-01,  5.94478844e-01,  2.97698400e-01,  6.14649063e-01,
            1.48462747e-01,  8.92689556e-01,  3.26585998e-01,  2.52467577e-01,
            5.68633720e-01,  5.71720695e-01,  2.68810802e-01,  1.40831765e-01,
            9.51883010e-02,  3.14545895e-01,  6.33206213e-01,  7.73649901e-01,
            3.45349387e-01,  9.06875508e-02,  1.06886418e-01,  5.20882806e-01,
            2.71384110e-01,  5.57708057e-02,  5.38179808e-01,  5.94934061e-01,
            1.02427299e+00,  1.01682492e+00,  1.07134838e+00,  6.65911915e-01,
            1.09660401e-01,  9.99551358e-02,  2.77988548e-01,  2.42159569e-01,
            6.15739148e-01,  2.28368164e-01,  5.19159262e-02,  5.32051913e-02,
            8.53960225e-01,  1.00894766e+00,  4.74878825e-01,  2.47348633e-02,
            7.04925540e-01,  3.93039192e-01,  6.15739148e-01,  7.47401910e-01,
            5.17521314e-01,  1.59446055e-01,  8.97437102e-02,  5.15318616e-01,
           -1.04107827e-01,  1.10218503e-01,  1.87586024e-01,  1.63247356e-01,
            4.66869086e-01,  8.54494372e-02,  1.06700748e-01,  1.45104601e-01,
            2.02976777e-01,  6.50404266e-01,  1.03562184e+00,  1.00886020e+00,
            2.37219075e-01,  6.44431926e-01,  1.33549562e-01,  4.69781872e-01,
            1.32766246e-01,  1.08667899e+00,  4.74136382e-01,  9.22600950e-01,
            6.15739148e-01,  4.05533216e-02,  5.93999813e-02,  7.87699524e-01,
            1.11249679e-01,  5.86839965e-01,  1.03996881e+00,  1.02875606e+00,
            2.25719763e-01,  9.97381884e-01,  1.04965429e+00,  9.76741677e-01,
            7.30650112e-01,  1.10439483e-01,  1.26788042e-01,  6.27852290e-01,
            7.70091888e-01,  1.36899148e-01,  9.96998782e-01,  8.77858384e-01,
            1.41551536e-01,  1.16217003e-01,  7.76144485e-01,  7.58727318e-01,
           -7.04647129e-02,  1.00317014e+00, -8.99446552e-02,  7.43720642e-01,
            5.39034881e-01,  1.05122728e+00,  5.46632069e-01,  3.69937263e-01,
            4.63431871e-01,  9.83814500e-02,  9.74778687e-01,  1.10439483e-01,
            4.30436791e-01,  9.73889874e-01,  1.30318443e-02,  3.82864040e-01,
            3.63231837e-01,  9.11507785e-01,  2.83862052e-01,  3.01194611e-01,
            2.37641895e-01,  8.13050878e-01,  7.20610564e-01,  5.73404593e-01,
            1.83322457e-01,  3.34517603e-02,  1.45892202e-01,  4.80431600e-01,
            8.00661532e-02,  8.91266472e-02,  1.06743340e-01,  1.18255788e-01,
            1.01411854e+00,  7.32165603e-01,  6.16430175e-01,  6.16430175e-01,
           -4.20158987e-02,  2.38038328e-01,  5.16451423e-01,  6.52868119e-02,
            6.56419480e-02,  9.53313789e-02,  7.63089024e-01,  6.12846155e-01,
            6.15739148e-01,  1.04117175e+00,  4.45042086e-01,  8.62196195e-02,
            1.63247356e-01,  5.77976666e-01,  6.28072636e-01,  9.52065482e-01,
            6.47630284e-01,  5.25710421e-01,  1.29615686e-01,  1.61792583e-01,
            9.92753352e-01,  7.58122150e-01,  8.74838994e-02,  8.90379502e-01,
            1.10439483e-01,  4.20450020e-01,  1.10515587e-01,  7.43720642e-01,
            1.09548199e-01,  8.49999762e-01,  3.73241199e-01,  1.50349161e-01,
           -6.57005419e-03,  9.95931305e-01,  6.27863334e-01,  1.44571641e-01,
            5.98964692e-01,  2.10829982e-01,  3.03475920e-01,  7.88747214e-01,
            4.70412618e-02,  1.22804718e-01,  5.92291993e-01,  6.66315158e-02,
            6.66635940e-01,  1.96583095e-01, -2.32225513e-02,  3.42713139e-01,
            1.50237544e-01,  4.92188546e-01,  1.10439483e-01,  1.04593192e-01,
            9.27419117e-01,  1.59446055e-01,  1.81536635e-02,  6.17328426e-01,
            6.94599343e-01,  8.03516565e-01,  2.72307013e-01,  7.25310016e-01,
            1.10439483e-01,  1.50012975e-01,  1.04551603e-01,  5.40581352e-01,
            1.07374434e-01,  1.06886418e-01,  7.46981809e-01,  8.72901760e-01,
            1.09660401e-01,  8.81395999e-02,  4.67819779e-01,  5.73404593e-01,
            6.66096534e-01,  1.69719363e-01,  3.00985353e-01,  1.00642538e+00,
            5.60676359e-01,  6.56335479e-01,  2.27698960e-01,  2.57255762e-01,
            6.21311750e-01,  1.64475945e-01,  5.32051913e-02,  7.89940800e-01,
            1.11574140e-01,  6.14192579e-01,  8.66696091e-01,  4.33154661e-01,
            6.43149322e-01,  3.35475911e-01,  1.53444166e-01,  7.69365011e-02,
            4.58274739e-01,  3.28433671e-01,  1.11249679e-01,  1.04661964e-01,
            2.54051551e-01,  9.30699673e-01,  6.53561636e-01,  1.09660401e-01,
            3.52269386e-01,  7.65845606e-02,  3.75032839e-01,  1.68311659e-01,
            1.11249679e-01,  4.79542529e-02,  1.59446055e-01,  3.06049228e-01,
            1.09525511e-01,  6.67703780e-01,  1.06886418e-01,  5.65618389e-02,
            6.76855472e-01,  8.21421742e-01,  6.60491155e-01,  4.98665666e-01,
            1.96583095e-01,  2.60050525e-02,  1.43074151e-01,  7.57763318e-01,
            6.42954302e-02,  1.59446055e-01, -1.58557540e-02,  4.22512587e-01,
            4.68116336e-01,  4.92188546e-01,  9.15864078e-01,  2.99598016e-01,
            9.96946392e-02,  1.48657339e-01,  7.69365011e-02,  1.47329056e-01,
            3.18143352e-01,  2.47958306e-01,  1.51692316e-01,  1.36489752e-01,
            7.99583366e-01,  1.38407253e-01,  9.53796039e-01,  1.33016602e-01,
            1.77889752e-01,  6.56181786e-01,  6.15083756e-01,  5.72728098e-01,
            1.09771020e+00,  5.16899793e-01,  7.47817054e-01,  4.67819779e-01,
            1.56286392e-01,  2.09971677e-01,  1.03882881e-01,  1.11249679e-01,
            4.21148273e-01,  7.86768231e-01,  1.31380588e-01,  3.69358645e-01,
            7.46001951e-01,  1.59435345e-01,  7.00761652e-01,  8.73294048e-02,
            1.02045318e+00,  1.45104601e-01,  1.06743340e-01,  8.89456734e-01,
            1.06767408e-01,  3.74414052e-02,  6.53561636e-01,  5.66359380e-01,
            4.70412618e-02,  1.62356102e-01,  8.69310584e-01,  1.06767408e-01,
            6.85111138e-02,  6.17830452e-01,  6.08860744e-01,  8.98467076e-01,
            3.69601945e-01,  1.02649673e+00,  1.36616756e-01,  9.92932165e-01,
            9.28768563e-01,  5.71204722e-01,  5.59649683e-01,  2.17792983e-01,
            3.30945318e-01,  2.58878379e-01,  7.98979486e-01,  2.86680512e-01,
            2.14010941e-02,  3.44367000e-01,  5.53243351e-01,  3.20602856e-01,
            1.10085203e-01,  1.41408458e-01,  6.50538772e-01,  2.71384110e-01,
            8.02706718e-01,  5.67559879e-01,  8.59175051e-01,  5.30638788e-01,
            1.09660401e-01,  4.11092475e-02,  2.85516401e-01,  1.11249679e-01,
            6.17328426e-01,  6.53813918e-02,  1.62437160e-01,  5.88837413e-01,
            1.06767408e-01,  8.71084247e-02,  7.81526528e-02,  7.42676841e-01,
            4.21320725e-01,  6.15739148e-01,  1.80579914e-01,  1.88614433e-01,
            7.55275682e-01,  8.70616817e-01,  5.68646225e-01,  8.74838994e-02,
            7.42940606e-01,  8.74607087e-01,  1.64609825e-01,  4.40486195e-01,
            1.14580360e-01,  1.03811254e+00,  1.80384397e-01,  2.31504605e-01,
            1.45914797e-01,  1.11249679e-01,  6.57240092e-02,  8.11138406e-01,
           -2.88645537e-03,  5.71539717e-01,  1.88905385e-01,  1.14061779e-02,
            8.31771915e-01, -1.13925609e-01,  1.06743340e-01,  3.07607989e-01,
            7.00677914e-01,  1.10439483e-01,  4.89244743e-01,  1.88093640e-02,
            4.27377141e-01,  1.95644356e-02,  9.88844441e-02,  4.79913902e-01,
            7.58923004e-01,  9.85842315e-01,  4.67712327e-01,  1.05729450e-01,
            5.71720695e-01,  1.09660401e-01,  7.08070409e-02,  7.68030701e-01,
           -1.62513385e-04,  5.84959632e-01,  8.33043428e-01,  2.86143361e-01,
            6.83324919e-02,  3.44348525e-01,  8.46609496e-02,  1.50661141e-01,
            1.81690757e-01,  2.86127463e-01,  1.14072707e-01,  1.06398001e+00,
            1.21994523e-01,  1.09570337e-01,  1.72951038e-01, -2.71289121e-02,
            4.51026304e-01,  3.11489805e-01,  5.89795249e-01,  7.89940800e-01,
            8.74838994e-02,  1.96118278e-01,  5.40999456e-01,  6.73258451e-02,
            1.56438661e-01,  1.04117175e+00,  6.30237887e-01,  1.98416638e-01,
            6.67703780e-01,  3.27246319e-01,  1.62437160e-01,  3.19311324e-01,
            1.08570315e-01,  6.43536661e-01,  1.10439483e-01,  8.45239768e-01,
            1.53647911e-01,  6.16163950e-01,  6.67792896e-01,  2.68620870e-01,
            1.10439483e-01,  5.20723881e-01,  3.12749650e-01,  3.17362212e-01,
            2.98352748e-01,  3.74131044e-02,  3.56370397e-01,  6.20503516e-02,
            8.28918614e-02,  2.22400713e-01,  3.01194611e-01,  1.09794906e-01,
            2.45868837e-02,  9.30822140e-01,  6.68294478e-01,  4.07410833e-01,
            3.19324955e-02,  2.66529493e-01,  1.59446055e-01,  1.67570103e-01,
            1.31470997e-01,  6.83619501e-01,  4.57014006e-01,  5.31605378e-01,
            6.17782050e-01,  5.22496506e-01,  1.63224761e-01,  3.54078318e-02,
            4.66653378e-02,  2.35511697e-01,  6.34709655e-02,  1.73173004e-01,
            1.67681720e-01,  1.11739603e+00,  4.11660926e-01,  7.50053518e-01,
            1.98416638e-01,  1.24075899e-01,  2.82939150e-01,  1.46855929e-01,
            1.88093640e-02,  6.15649084e-01,  3.10753143e-01,  2.82332334e-02,
            1.05749155e+00,  4.26925792e-01,  6.72444454e-01,  1.26948219e-01,
            4.78512686e-02,  2.51272620e-01,  6.93377314e-01,  3.86483155e-01,
            1.08925030e+00,  3.70232540e-02,  1.02558048e+00,  4.67819779e-01,
            3.09192999e-01,  1.12403227e-01,  1.45201580e-01,  1.61116186e-01,
            1.01586343e+00,  7.86444588e-01,  1.39564222e-01,  8.09073080e-02,
            9.30990611e-01,  9.53844823e-02,  2.49196934e-01,  1.56756620e-01,
            4.31489359e-01,  1.60557193e-01,  6.51849647e-01,  6.15671751e-01,
            2.58235621e-01,  5.60083010e-01,  1.06439749e+00,  2.46347671e-01,
            1.59446055e-01,  3.12749650e-01,  3.12749650e-01,  1.38359359e-01,
            4.36351045e-01,  5.65269634e-01,  1.10439483e-01,  1.10439483e-01,
            4.73382061e-01,  3.97845622e-01,  9.59409118e-01,  9.04359237e-02,
            9.32614191e-02,  1.68644568e-01,  1.19012742e-01,  7.78385760e-01,
            4.77709516e-01,  9.23278419e-02,  8.52202769e-01,  2.29235093e-01,
            8.93681893e-02,  1.39327082e-01,  6.29370229e-01,  3.44751108e-01,
            1.09906523e-01,  3.36294469e-01,  7.65845606e-02,  9.99492366e-01,
            1.36616756e-01,  3.10588913e-02,  1.52987565e-01,  8.79319946e-01,
            1.79125142e-01,  8.16732146e-01,  4.86100638e-01,  6.01295349e-01,
            8.55130714e-02,  8.97400287e-02,  1.41365866e-01, -5.33188988e-03,
            6.13118999e-01,  1.06743340e-01,  5.35440050e-01,  1.67435598e-01,
            1.09660401e-01,  7.68978096e-01,  1.09593004e-01,  9.55625483e-01,
            6.99508950e-01,  9.98624949e-01,  4.67604541e-01,  3.99828714e-02,
            1.23062009e-01,  1.24218977e-01,  6.72124453e-01,  8.39661905e-02,
            1.90078411e-01,  4.14198891e-01,  1.09660401e-01,  3.84155572e-01,
            4.35614035e-01,  4.74428517e-01,  1.27772042e-01,  2.20309336e-01,
            8.51496136e-01,  6.03184600e-01,  9.52123685e-02,  5.52150043e-01,
            2.49196934e-01,  7.02197025e-01,  5.17679190e-01,  2.45077500e-01,
            1.11065680e-01,  9.24623473e-02,  2.47937089e-01,  6.73648850e-01,
            2.20309336e-01,  9.02902989e-01,  1.21950516e-01,  9.58826291e-02,
            2.34145684e-01,  5.88537281e-01,  9.68058794e-02,  3.11489805e-01,
            6.45560324e-01,  2.11800442e-01,  1.54149637e-02,  7.79389822e-02,
            7.91538741e-01,  1.20114555e-01,  2.53714609e-01,  6.13985928e-01,
            1.32801201e-01,  1.05140141e-01,  1.98416638e-01,  4.42735611e-01,
            1.09660401e-01,  8.28305259e-01,  6.43741475e-01,  3.55250357e-01,
            1.06767408e-01,  1.39106102e-01,  1.70309535e-01,  8.61182482e-01,
            1.54779673e-01,  1.11249679e-01,  1.75306559e-01,  4.72923605e-01,
            1.56814135e-01,  3.43918557e-01,  9.69402533e-01,  6.44447136e-02,
            1.77889752e-01,  2.76850400e-02, -7.04647129e-02,  6.99968458e-02,
            2.97806431e-01,  9.55840332e-01,  9.40135731e-02, -1.55970998e-01,
            6.52874875e-01,  1.03339007e+00,  6.54541495e-01,  6.53544651e-01,
            8.58503928e-01,  3.34936827e-01,  6.22653744e-01,  1.06767408e-01,
           -5.68617215e-02,  2.45039636e-01,  8.57273656e-01,  4.35614035e-01,
            3.06972131e-01,  7.09055524e-01,  7.24601181e-01,  4.79975750e-01,
            1.18255788e-01,  1.61923272e-01,  1.21994523e-01,  7.93652938e-01,
            3.67073105e-01,  6.18274544e-03,  7.43353550e-01,  6.87929843e-01,
            1.66003539e-01,  1.62437160e-01,  1.10439483e-01,  8.36886921e-01,
            8.10349906e-01,  8.15518851e-02,  6.63356848e-01,  2.74588321e-01,
            1.23062009e-01,  5.38384602e-01,  2.89639572e-01,  1.03145848e+00,
            5.46061619e-01,  4.84937100e-01,  8.65503223e-02])



선형 회귀 모델의 RMSE는 이러하다.


```python
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(train_target, train_data_prediction)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    np.float64(0.3790022086226017)




```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(train_data, train_target)
```




<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-8 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;...
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;decisiontreeregressor&#x27;,
                 DecisionTreeRegressor(random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-92" type="checkbox" ><label for="sk-estimator-id-92" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;...
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;decisiontreeregressor&#x27;,
                 DecisionTreeRegressor(random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-93" type="checkbox" ><label for="sk-estimator-id-93" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>columntransformer: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                             SimpleImputer(strategy=&#x27;median&#x27;)),
                                            (&#x27;standardscaler&#x27;,
                                             StandardScaler())]),
                  transformers=[(&#x27;log&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                      func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-94" type="checkbox" ><label for="sk-estimator-id-94" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-95" type="checkbox" ><label for="sk-estimator-id-95" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-96" type="checkbox" ><label for="sk-estimator-id-96" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log1p</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;, func=&lt;ufunc &#x27;log1p&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-97" type="checkbox" ><label for="sk-estimator-id-97" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-98" type="checkbox" ><label for="sk-estimator-id-98" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-99" type="checkbox" ><label for="sk-estimator-id-99" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-100" type="checkbox" ><label for="sk-estimator-id-100" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-101" type="checkbox" ><label for="sk-estimator-id-101" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Pclass&#x27;, &#x27;Age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-102" type="checkbox" ><label for="sk-estimator-id-102" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-103" type="checkbox" ><label for="sk-estimator-id-103" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-104" type="checkbox" ><label for="sk-estimator-id-104" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeRegressor.html">?<span>Documentation for DecisionTreeRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(random_state=42)</pre></div> </div></div></div></div></div></div>




```python
train_data_prediction = tree_reg.predict(train_data)
train_data_prediction
```




    array([0.        , 1.        , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 0.        , 0.5       , 0.        , 1.        ,
           0.        , 0.33333333, 1.        , 1.        , 0.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 0.875     , 0.        , 0.        ,
           0.        , 0.16666667, 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 0.        , 0.875     , 0.        , 0.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           1.        , 1.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.5       , 0.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.        , 0.        ,
           1.        , 0.1       , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.5       ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 0.        , 0.5       ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 1.        , 0.5       , 0.        , 0.        ,
           1.        , 0.        , 1.        , 1.        , 1.        ,
           1.        , 0.1       , 0.        , 0.875     , 0.5       ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 1.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 1.        , 0.        ,
           0.5       , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.5       , 0.        , 0.        , 0.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 0.        , 1.        , 0.5       , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           0.1       , 1.        , 0.        , 0.        , 0.875     ,
           0.        , 0.        , 0.5       , 1.        , 1.        ,
           0.        , 1.        , 1.        , 0.        , 0.875     ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 0.5       , 0.        ,
           0.        , 1.        , 0.        , 0.5       , 1.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           0.875     , 1.        , 0.        , 1.        , 0.        ,
           1.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 1.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           1.        , 0.        , 1.        , 1.        , 1.        ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           1.        , 1.        , 0.5       , 0.        , 0.        ,
           0.5       , 1.        , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 1.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 0.875     , 1.        ,
           1.        , 0.        , 0.5       , 0.        , 0.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.33333333, 1.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.        , 1.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.5       , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 1.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 0.1       , 1.        ,
           0.66666667, 1.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 0.5       , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 0.        ,
           1.        , 0.        , 1.        , 1.        , 0.1       ,
           1.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 1.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 1.        , 0.66666667, 0.        , 1.        ,
           1.        , 0.        , 1.        , 1.        , 0.        ,
           0.        , 1.        , 0.        , 1.        , 0.        ,
           1.        , 0.        , 0.        , 1.        , 0.16666667,
           0.        , 1.        , 0.        , 0.5       , 0.        ,
           1.        , 0.16666667, 0.        , 1.        , 0.        ,
           1.        , 0.5       , 1.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 1.        , 0.        , 1.        , 1.        ,
           0.1       , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.16666667, 1.        ,
           1.        , 1.        , 1.        , 0.875     , 0.5       ,
           0.        , 1.        , 1.        , 0.        , 0.33333333,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           1.        , 0.        , 1.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 1.        ,
           0.        , 0.        , 1.        , 0.1       , 0.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 0.33333333, 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 0.5       , 1.        ,
           1.        , 0.        , 1.        , 0.        , 1.        ,
           0.        , 1.        , 0.        , 1.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.5       , 0.        , 1.        ,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 1.        ,
           0.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           1.        , 1.        , 0.5       , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 0.66666667, 1.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 0.33333333, 0.        , 1.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 0.        , 1.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           1.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.1       , 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 0.        , 1.        , 0.        ,
           0.1       , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 0.        , 0.        ,
           0.33333333, 1.        , 1.        , 1.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 1.        , 0.        ,
           0.        , 0.5       , 1.        , 0.1       , 1.        ,
           1.        , 1.        , 0.16666667, 0.        , 0.        ,
           1.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 0.        ,
           1.        , 1.        , 1.        , 1.        , 0.16666667,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           1.        , 1.        , 0.        , 0.        , 1.        ,
           0.        , 1.        , 0.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 0.        , 1.        ,
           1.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.        , 0.        , 1.        ,
           0.        ])




```python
tree_mse = mean_squared_error(train_target, train_data_prediction)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    np.float64(0.10911506778263545)



결정 트리 모델의 RMSE는 선형 회귀 모델보다 낮게 나온다.


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(n_estimators=100, random_state=42))

forest_reg.fit(train_data, train_target)
```




<style>#sk-container-id-9 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-9 {
  color: var(--sklearn-color-text);
}

#sk-container-id-9 pre {
  padding: 0;
}

#sk-container-id-9 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-9 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-9 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-9 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-9 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-9 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-9 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-9 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-9 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-9 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-9 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-9 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-9 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-9 div.sk-label label.sk-toggleable__label,
#sk-container-id-9 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-9 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-9 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-9 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-9 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-9 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-9 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-9 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-9 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-9 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;...
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;randomforestregressor&#x27;,
                 RandomForestRegressor(random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-105" type="checkbox" ><label for="sk-estimator-id-105" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                              SimpleImputer(strategy=&#x27;median&#x27;)),
                                                             (&#x27;standardscaler&#x27;,
                                                              StandardScaler())]),
                                   transformers=[(&#x27;log&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                                       func=&lt;ufunc &#x27;log1p&#x27;...
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])),
                (&#x27;randomforestregressor&#x27;,
                 RandomForestRegressor(random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-106" type="checkbox" ><label for="sk-estimator-id-106" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>columntransformer: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                             SimpleImputer(strategy=&#x27;median&#x27;)),
                                            (&#x27;standardscaler&#x27;,
                                             StandardScaler())]),
                  transformers=[(&#x27;log&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;,
                                                                      func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 [&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;)])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-107" type="checkbox" ><label for="sk-estimator-id-107" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Parch&#x27;, &#x27;Fare&#x27;, &#x27;SibSp&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-108" type="checkbox" ><label for="sk-estimator-id-108" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-109" type="checkbox" ><label for="sk-estimator-id-109" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>log1p</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(feature_names_out=&#x27;one-to-one&#x27;, func=&lt;ufunc &#x27;log1p&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-110" type="checkbox" ><label for="sk-estimator-id-110" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-111" type="checkbox" ><label for="sk-estimator-id-111" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x7f625e7105d0&gt;</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-112" type="checkbox" ><label for="sk-estimator-id-112" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-113" type="checkbox" ><label for="sk-estimator-id-113" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-114" type="checkbox" ><label for="sk-estimator-id-114" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Pclass&#x27;, &#x27;Age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-115" type="checkbox" ><label for="sk-estimator-id-115" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-116" type="checkbox" ><label for="sk-estimator-id-116" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-117" type="checkbox" ><label for="sk-estimator-id-117" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(random_state=42)</pre></div> </div></div></div></div></div></div>




```python
train_data_prediction = forest_reg.predict(train_data)
train_data_prediction
```




    array([0.04133333, 1.        , 0.84      , 1.        , 0.        ,
           0.06      , 0.11      , 0.09      , 0.83      , 1.        ,
           0.9       , 0.92      , 0.10333333, 0.03      , 0.25      ,
           0.93      , 0.08      , 0.38846429, 0.45      , 0.93      ,
           0.00666667, 0.32011905, 0.86      , 0.86      , 0.09      ,
           0.61      , 0.        , 0.12      , 0.97      , 0.        ,
           0.1       , 0.94      , 0.87850123, 0.16      , 0.072     ,
           0.23      , 0.16462302, 0.        , 0.11      , 0.71      ,
           0.04      , 0.39      , 0.        , 0.97      , 0.89      ,
           0.        , 0.03      , 0.87850123, 0.16      , 0.11      ,
           0.        , 0.32      , 1.        , 0.99      , 0.03      ,
           0.86      , 0.98375   , 0.0628373 , 0.99      , 0.        ,
           0.19      , 1.        , 0.03      , 0.04      , 0.07933333,
           0.88      , 1.        , 0.215     , 0.81      , 0.0775    ,
           0.09      , 0.02      , 0.0625    , 0.06      , 0.90166667,
           0.        , 0.        , 0.        , 1.        , 0.73      ,
           0.        , 0.67      , 0.87034238, 0.        , 0.99      ,
           0.78      , 0.03      , 0.        , 0.99      , 0.        ,
           0.12      , 0.01      , 0.1       , 0.1       , 0.        ,
           0.        , 0.09      , 0.74      , 0.98      , 0.02      ,
           0.3       , 0.        , 0.21      , 0.        , 0.18083333,
           0.        , 0.81      , 0.39670076, 0.        , 0.63      ,
           0.04      , 0.27      , 0.        , 0.06      , 0.14      ,
           0.02      , 0.        , 0.04      , 0.1       , 0.2       ,
           0.01      , 0.        , 0.13      , 1.        , 0.06      ,
           0.74      , 0.11736106, 0.66      , 0.88      , 0.12      ,
           0.        , 0.12      , 0.04      , 0.99      , 0.        ,
           0.03      , 1.        , 0.3       , 0.00916667, 0.21      ,
           0.27      , 0.93      , 0.75      , 0.04      , 0.045     ,
           0.055     , 0.69      , 0.03      , 0.        , 0.45628571,
           0.        , 1.        , 0.04      , 0.01      , 0.        ,
           0.08      , 0.92      , 0.01      , 0.13      , 0.        ,
           0.03      , 0.98      , 0.1535    , 0.        , 0.06      ,
           0.88      , 1.        , 0.        , 0.08433333, 0.44683333,
           0.        , 0.03      , 0.85      , 0.02      , 0.09      ,
           0.15      , 0.09      , 0.39      , 0.04      , 0.02      ,
           0.01      , 0.13333333, 0.04      , 0.71      , 0.89      ,
           0.00333333, 0.96      , 0.43654762, 0.1       , 0.        ,
           1.        , 0.11083333, 0.86      , 0.97      , 0.92      ,
           0.97      , 0.11736106, 0.02      , 0.87850123, 0.52258333,
           0.03      , 0.        , 0.01      , 0.03      , 0.8125    ,
           0.38      , 0.14      , 0.68      , 0.87      , 0.56      ,
           0.08      , 0.95      , 0.        , 0.04      , 0.0740372 ,
           1.        , 0.88      , 0.11380952, 1.        , 0.05      ,
           0.47659524, 0.07      , 0.04      , 0.        , 0.73      ,
           0.        , 0.42      , 0.03      , 0.015     , 0.        ,
           1.        , 0.22333333, 0.03      , 0.66      , 0.01      ,
           0.2       , 0.02266667, 1.        , 0.42      , 0.05033333,
           0.12      , 0.96      , 0.13      , 0.12      , 0.01      ,
           0.15      , 0.28394231, 0.93166667, 0.79      , 0.        ,
           0.00596825, 0.11      , 0.01      , 0.04      , 0.27      ,
           0.7       , 1.        , 1.        , 1.        , 0.88      ,
           0.11736106, 0.56      , 0.14      , 0.01      , 0.87850123,
           0.01      , 0.01      , 0.457     , 0.96      , 1.        ,
           0.18      , 0.66      , 1.        , 0.2       , 0.87850123,
           1.        , 0.02637363, 0.        , 0.        , 0.82      ,
           0.02      , 0.03      , 0.00916667, 0.499     , 0.09933333,
           0.        , 0.68      , 0.        , 0.45628571, 0.93      ,
           0.99      , 1.        , 0.04666667, 0.06      , 0.        ,
           0.07933333, 0.09503571, 0.23      , 0.88      , 0.94      ,
           0.87850123, 0.64      , 0.14      , 1.        , 0.        ,
           0.96      , 1.        , 1.        , 0.02      , 1.        ,
           1.        , 0.99      , 0.37      , 0.        , 0.03980952,
           0.9       , 0.9625    , 0.04      , 0.92      , 1.        ,
           0.        , 0.16      , 1.        , 0.85      , 0.        ,
           1.        , 0.02      , 1.        , 0.8       , 1.        ,
           0.84      , 0.03416667, 0.24      , 0.0625    , 1.        ,
           0.        , 0.212     , 1.        , 0.61      , 0.17      ,
           0.98      , 0.99      , 0.38846429, 0.        , 0.09278571,
           0.52258333, 0.89      , 0.95      , 1.        , 0.04      ,
           0.        , 0.31      , 0.066     , 0.07      , 0.        ,
           0.03      , 1.        , 0.38      , 0.97      , 0.97      ,
           0.02      , 0.        , 0.05      , 0.        , 0.03      ,
           0.01      , 1.        , 0.93      , 0.87850123, 1.        ,
           0.94      , 0.04      , 0.499     , 0.09      , 0.27      ,
           1.        , 0.99      , 0.16      , 0.11      , 0.01      ,
           1.        , 0.98      , 0.34266667, 1.        , 0.        ,
           0.0425    , 0.1       , 1.        , 0.        , 0.979     ,
           0.78      , 0.67      , 0.06      , 1.        , 0.84      ,
           0.19      , 0.24      , 0.        , 0.01      , 1.        ,
           0.45383333, 0.03      , 0.01      , 0.04      , 0.12      ,
           0.        , 0.        , 0.99      , 0.11      , 0.        ,
           0.        , 0.03      , 1.        , 0.        , 0.745     ,
           0.03      , 1.        , 0.934     , 0.04      , 0.13      ,
           0.        , 0.03      , 0.33      , 0.08      , 0.05      ,
           0.00596825, 0.98      , 1.        , 0.11736106, 0.59416667,
           0.65702381, 0.95      , 0.82      , 0.02      , 0.34      ,
           1.        , 0.        , 0.91      , 0.14      , 0.14      ,
           0.87      , 0.        , 0.457     , 0.995     , 0.64      ,
           0.95      , 0.99      , 0.9425    , 0.89      , 0.7       ,
           0.01      , 0.01      , 0.00666667, 0.76      , 0.        ,
           0.65      , 0.01      , 1.        , 0.93      , 0.11736106,
           0.74916667, 0.0225    , 0.03      , 0.02      , 0.        ,
           0.        , 0.        , 0.085     , 0.        , 0.95      ,
           0.00596825, 0.        , 0.99      , 0.96666667, 0.05      ,
           0.02333333, 0.        , 0.11      , 0.        , 0.95      ,
           0.        , 0.        , 0.07      , 0.64      , 0.7       ,
           0.        , 1.        , 0.05      , 0.01      , 0.93      ,
           0.01      , 0.03      , 0.19      , 0.06      , 0.        ,
           0.        , 0.99      , 0.02      , 0.43      , 0.05      ,
           0.        , 0.43      , 0.14      , 0.18      , 1.        ,
           0.13      , 1.        , 0.65702381, 0.06666667, 0.77364286,
           0.69097869, 0.        , 0.96      , 0.99      , 0.04      ,
           0.03      , 1.        , 0.02819048, 1.        , 0.10683333,
           1.        , 0.        , 0.        , 0.99      , 0.16462302,
           0.02      , 0.93      , 0.0375    , 0.45383333, 0.04      ,
           0.78      , 0.16462302, 0.06      , 0.89      , 0.19      ,
           1.        , 0.43654762, 1.        , 0.        , 1.        ,
           1.        , 0.01      , 0.        , 0.57      , 0.21      ,
           0.03      , 0.99      , 0.65583333, 0.01      , 0.86      ,
           0.74      , 0.03      , 0.02666667, 0.65      , 0.78      ,
           0.01      , 1.        , 0.0375    , 1.        , 0.83      ,
           0.11736106, 0.015     , 0.3225    , 0.        , 0.03      ,
           0.05      , 0.01      , 0.36      , 0.16462302, 0.6875    ,
           0.68      , 0.99      , 0.99      , 0.87850123, 0.47659524,
           0.0325    , 1.        , 1.        , 0.09      , 0.34266667,
           0.97      , 1.        , 0.        , 0.11      , 0.06      ,
           1.        , 0.        , 0.54      , 0.        , 0.        ,
           0.        , 0.99      , 0.        , 0.31263858, 0.        ,
           0.        , 1.        , 0.01      , 0.        , 0.87      ,
           0.96      , 0.        , 0.01      , 0.03      , 0.93583333,
           0.01      , 0.16166667, 0.99      , 1.        , 0.97      ,
           0.07      , 0.        , 0.96      , 0.11736106, 0.        ,
           1.        , 0.06416667, 0.33      , 0.99      , 0.045     ,
           0.09      , 0.72      , 0.69      , 0.03      , 0.01      ,
           0.        , 0.        , 1.        , 0.        , 0.        ,
           0.73      , 0.        , 0.88      , 0.        , 0.01      ,
           0.995     , 0.34266667, 0.14      , 0.01      , 0.        ,
           0.01      , 1.        , 0.1       , 0.44683333, 0.95      ,
           0.73      , 0.01      , 0.59      , 0.        , 0.81      ,
           0.        , 0.98      , 0.        , 0.98      , 0.32      ,
           0.02      , 0.        , 0.28      , 0.        , 0.09      ,
           0.68      , 0.        , 0.04583333, 0.        , 0.65      ,
           0.09      , 0.        , 0.39670076, 0.025     , 1.        ,
           1.        , 0.11      , 0.09      , 0.67      , 0.        ,
           0.01      , 0.        , 0.82      , 0.02      , 0.88      ,
           0.02      , 0.75      , 0.        , 0.01      , 0.02      ,
           0.02      , 0.01      , 0.03166667, 0.16      , 1.        ,
           0.81      , 0.93      , 0.44683333, 0.11      , 0.015     ,
           0.        , 0.03      , 0.98963781, 0.23      , 0.01      ,
           1.        , 0.96666667, 0.22      , 0.00560531, 0.04333333,
           0.        , 0.98      , 0.886     , 0.94      , 0.88      ,
           1.        , 0.65702381, 0.78      , 0.27      , 0.01      ,
           0.01      , 1.        , 0.97      , 0.01      , 0.01      ,
           1.        , 0.        , 0.32011905, 0.01      , 0.79      ,
           0.005     , 0.89      , 0.98963781, 0.01      , 0.11      ,
           1.        , 0.2       , 0.        , 0.        , 0.        ,
           0.        , 0.01      , 0.85      , 0.        , 0.        ,
           0.86      , 0.12      , 0.99      , 0.01      , 0.70797619,
           0.13      , 0.06      , 1.        , 0.15      , 0.07090909,
           0.95      , 0.98      , 0.02      , 0.        , 1.        ,
           1.        , 0.21257576, 0.045     , 0.0225    , 1.        ,
           0.        , 0.        , 0.67      , 0.99      , 0.03183333,
           0.99      , 0.06      , 0.1696606 , 0.04285714, 0.15166667,
           0.        , 0.09      , 0.42      , 0.        , 0.9       ,
           0.01664835, 0.11736106, 0.96      , 0.        , 1.        ,
           0.85      , 1.        , 0.34      , 0.11      , 0.04      ,
           0.02      , 0.9       , 0.03      , 1.        , 0.02      ,
           0.11736106, 0.        , 0.01      , 0.31      , 0.        ,
           0.01      , 0.99      , 0.65      , 0.02358333, 0.16      ,
           0.32011905, 0.97      , 0.92      , 0.99      , 0.7       ,
           0.06      , 0.01      , 0.33      , 0.01      , 1.        ,
           0.        , 0.        , 0.02      , 0.14      , 0.01      ,
           0.        , 0.23      , 0.07      , 0.01      , 0.        ,
           1.        , 0.72      , 0.02      , 0.87      , 0.03      ,
           0.04      , 0.44683333, 0.98      , 0.11736106, 0.99      ,
           0.63      , 1.        , 0.16462302, 0.        , 0.14      ,
           1.        , 0.01      , 0.        , 0.90166667, 0.835     ,
           0.09      , 0.09166667, 1.        , 0.        , 0.        ,
           0.01      , 0.        , 0.        , 0.04033333, 1.        ,
           0.07      , 0.        , 0.18      , 1.        , 0.4       ,
           0.84      , 0.99      , 0.805     , 0.82      , 0.16462302,
           0.04633333, 0.065     , 0.99      , 0.01      , 0.        ,
           0.95      , 0.85      , 0.03      , 0.03      , 0.99      ,
           0.        , 1.        , 0.09      , 0.06      , 0.99      ,
           0.94      , 0.        , 0.01      , 0.        , 0.98      ,
           0.94166667, 0.        , 0.12      , 0.0975    , 0.04      ,
           0.06      , 0.07      , 1.        , 0.28      , 0.97      ,
           0.17090909])



랜덤 포레스트 회귀 모델의 RMSE는 선형 회귀 모델보다는 좋게 보인다.


```python
forest_mse = mean_squared_error(train_target, train_data_prediction)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    np.float64(0.16669064310924084)



전체적으로 RMSE가 너무 낮아 과대 적합 현상이 있는 듯하다. 교차 검증으로 모델 간 성능 차이를 더 확실히 구분해 본다.


```python
from sklearn.model_selection import cross_val_score

lin_rmses = -cross_val_score(lin_reg, train_data, train_target,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383078</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.020555</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.331735</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.376683</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.390362</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.394982</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.400961</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
tree_rmses = -cross_val_score(tree_reg, train_data, train_target,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.452381</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.042742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.402624</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.415701</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.450413</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.469939</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.540982</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
forest_rmses = -cross_val_score(forest_reg, train_data, train_target,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(forest_rmses).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.372702</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.040384</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.323030</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.333821</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.374477</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.401408</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.436081</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



교차 검증을 활용한 RMSE는 세 모델 모두 이전보다 높게 나온다. 특히 결정 트리 모델의 RMSE가 크게 증가했다.

최종 모델에는 평균적인 RMSE가 가장 낮은 랜덤 포레스트 회귀 모델을 사용하기로 한다.


```python
final_prediction = forest_reg.predict(test)
```


```python
final_prediction
```




    array([0.11      , 0.14      , 0.65      , 0.72      , 0.45      ,
           0.00916667, 0.05      , 0.11      , 0.94      , 0.05      ,
           0.        , 0.08583333, 1.        , 0.33      , 1.        ,
           0.9525    , 0.03583333, 0.67      , 0.69      , 0.13      ,
           0.27      , 0.776     , 1.        , 0.48      , 0.93      ,
           0.01      , 1.        , 0.65      , 0.72      , 0.16      ,
           0.        , 0.03      , 0.75      , 0.28      , 0.71866667,
           0.32      , 0.03      , 0.04      , 0.        , 0.44683333,
           0.025     , 0.65702381, 0.085     , 1.        , 0.99      ,
           0.02      , 0.18316667, 0.11736106, 0.99      , 0.65      ,
           0.62      , 0.07      , 0.93      , 0.87      , 0.08333333,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.04833333, 0.5325    , 0.01664835, 0.95      , 0.53      ,
           0.9       , 0.7       , 0.03      , 0.12      , 0.93      ,
           0.695     , 0.01      , 0.4       , 0.07933333, 1.        ,
           0.28      , 0.        , 0.69      , 0.04      , 0.695     ,
           0.97      , 0.2       , 0.27      , 0.        , 0.0975    ,
           0.07      , 0.98      , 0.69      , 0.87850123, 0.99      ,
           0.24      , 0.39670076, 0.88      , 0.        , 0.20766667,
           0.14933333, 1.        , 0.62      , 0.69      , 0.02      ,
           1.        , 0.04      , 0.11736106, 0.1535    , 0.82      ,
           0.06666667, 0.24      , 0.11736106, 0.06      , 0.015     ,
           0.07883333, 0.77492571, 1.        , 0.93      , 1.        ,
           0.165     , 0.02      , 0.93      , 0.31666667, 0.99      ,
           1.        , 0.0740372 , 1.        , 0.00596825, 0.11736106,
           0.43      , 0.19      , 0.84      , 0.45628571, 0.        ,
           0.59416667, 0.37      , 0.14      , 0.02      , 0.09      ,
           0.        , 0.16      , 0.0475    , 0.05      , 0.03      ,
           0.05      , 0.88      , 0.25      , 0.04985714, 0.72604762,
           0.26      , 0.02333333, 0.        , 0.65702381, 0.05      ,
           1.        , 0.        , 0.08      , 0.59      , 0.01      ,
           0.04      , 1.        , 0.77      , 0.72604762, 0.6       ,
           0.98963781, 0.95      , 0.92      , 0.1       , 0.28014286,
           0.68      , 0.48      , 0.01      , 0.99      , 0.05      ,
           0.        , 0.36      , 0.1675    , 0.16462302, 0.03      ,
           1.        , 1.        , 0.25      , 1.        , 0.99      ,
           0.04      , 0.38      , 1.        , 0.11736106, 0.97      ,
           0.01      , 0.88      , 0.47209524, 0.01      , 0.035     ,
           0.02      , 0.09933333, 0.8       , 0.37      , 0.86      ,
           0.15      , 0.94      , 0.33      , 0.01      , 0.02      ,
           0.9       , 0.99      , 0.37      , 0.94      , 0.01      ,
           0.08433333, 0.18147879, 0.01      , 0.99      , 0.        ,
           0.50333333, 0.        , 0.1025    , 0.57      , 0.08      ,
           0.1       , 0.97      , 0.33      , 0.97      , 0.        ,
           0.99      , 0.        , 0.89958333, 0.67      , 0.93      ,
           0.27      , 0.04      , 0.87850123, 0.        , 0.09278571,
           0.1125    , 1.        , 0.16      , 0.        , 0.31      ,
           0.01      , 0.26      , 0.8       , 0.914     , 1.        ,
           0.88      , 0.93      , 0.43      , 0.        , 0.11      ,
           0.25      , 0.87666667, 0.07380952, 0.99      , 0.12      ,
           0.66      , 0.09      , 0.52      , 0.        , 0.205     ,
           0.        , 0.11736106, 0.        , 0.884     , 0.11      ,
           0.02      , 0.03      , 0.99      , 0.95      , 0.05      ,
           0.        , 0.        , 0.        , 0.03      , 0.        ,
           0.14666667, 0.11736106, 0.94      , 0.96      , 0.        ,
           0.99      , 0.0975    , 0.01      , 0.02      , 0.03166667,
           0.04      , 1.        , 0.87850123, 0.18      , 0.79      ,
           0.        , 0.00596825, 0.25      , 0.16462302, 0.        ,
           0.06      , 0.18      , 0.16462302, 0.14      , 0.02      ,
           0.        , 0.62      , 0.16      , 0.        , 0.39      ,
           0.18      , 0.13333333, 0.09230952, 0.03      , 0.87850123,
           0.9       , 0.46      , 0.98      , 0.29      , 0.04      ,
           0.03      , 0.65      , 0.        , 0.08437363, 0.96      ,
           0.87      , 0.39      , 0.42      , 0.19      , 0.02      ,
           0.1535    , 0.0960873 , 0.0475    , 0.9325    , 1.        ,
           0.06      , 1.        , 0.02      , 0.06      , 0.005     ,
           0.97      , 0.32      , 0.        , 0.43      , 0.16      ,
           0.11666667, 0.5875    , 0.051     , 0.13916667, 0.16462302,
           0.16916667, 0.15      , 0.08883333, 0.99      , 0.56      ,
           0.95      , 0.0475    , 0.18      , 0.07      , 0.95      ,
           0.99      , 0.01      , 0.0425    , 0.13      , 0.91      ,
           0.25      , 1.        , 0.        , 0.11736106, 0.73      ,
           0.08      , 1.        , 0.95      , 0.72      , 0.97      ,
           0.36      , 0.07      , 0.31      , 0.99      , 0.52      ,
           0.065     , 1.        , 0.15      , 0.32647619, 1.        ,
           0.99      , 0.08      , 0.01166667, 0.15      , 0.17      ,
           0.11736106, 0.        , 0.17      , 0.19      , 0.36179762,
           1.        , 0.081     , 0.01      , 0.06789835, 0.08      ,
           0.3       , 0.96      , 0.22      , 0.07      , 0.19      ,
           1.        , 0.09      , 1.        , 0.025     , 0.03      ,
           1.        , 0.        , 1.        , 0.39      , 0.175     ,
           0.20666667, 0.0375    , 0.31      , 0.98963781, 0.78      ,
           0.87850123, 1.        , 0.77492571, 0.        , 1.        ,
           0.        , 0.        , 0.19      ])



테스트 셋으로 예측을 진행한 결과를 그대로 쓸 수는 없다. 제출해야 하는 데이터는 생존 여부를 정수 0 혹은 1로 표기해야 하기 때문이다.

모델이 1에 가깝게 예측할수록 생존했을 가능성이 높다고 결론을 내린 것이므로, 0.9 정도를 기준으로 하여 1과 0만으로 데이터를 변환한다.


```python
final_prediction[(final_prediction >= 0.9)] = 1
final_prediction[(final_prediction < 0.9)] = 0
final_prediction
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0.])



데이터프레임으로 변환하여 표기하면 이러하다.


```python
pd.DataFrame(final_prediction, columns=['Survived'])
```





  <div id="df-5a421f45-a6f8-47f4-9a4e-4d87566cb5d2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 1 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5a421f45-a6f8-47f4-9a4e-4d87566cb5d2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5a421f45-a6f8-47f4-9a4e-4d87566cb5d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5a421f45-a6f8-47f4-9a4e-4d87566cb5d2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7ea6b73b-0127-4722-af47-1ad08d70bd91">
  <button class="colab-df-quickchart" onclick="quickchart('df-7ea6b73b-0127-4722-af47-1ad08d70bd91')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7ea6b73b-0127-4722-af47-1ad08d70bd91 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




아까 따로 저장해둔 PassengerId와 합쳐 예측 결과값을 완성한다.


```python
final_submission = pd.concat([PassengerId, pd.DataFrame(final_prediction, columns=['Survived'], dtype=np.int64)], axis=1)
final_submission
```





  <div id="df-629943dd-c4f6-4c42-9943-81bf0b6543e3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 2 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-629943dd-c4f6-4c42-9943-81bf0b6543e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-629943dd-c4f6-4c42-9943-81bf0b6543e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-629943dd-c4f6-4c42-9943-81bf0b6543e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-51520099-068a-48ca-9bbc-0e83349ccc57">
  <button class="colab-df-quickchart" onclick="quickchart('df-51520099-068a-48ca-9bbc-0e83349ccc57')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-51520099-068a-48ca-9bbc-0e83349ccc57 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_29ab25b2-0e11-4e26-be10-37e82eb883f3">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('final_submission')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_29ab25b2-0e11-4e26-be10-37e82eb883f3 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('final_submission');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
final_submission.to_csv("/content/drive/MyDrive/Colab Notebooks/Files/final_submission.csv", index = False)
```


```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/Colab Notebooks/Untitled1.ipynb"
```

    [NbConvertApp] WARNING | pattern '/content/drive/MyDrive/Colab Notebooks/Untitled1.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

