# 추천 시스템

- 한번의 거래 (transaction)
  - 사람 _user id_
  - 아이템 _item id_


- 규칙기반 rule-based
  - 연관규칙분석 _by A Priori Algorithm_
- 내용기반 content-based
  - 데이터간 거리 기반
    - 내적
    - 피어슨 유사도
    - 코사인 유사도
    - 유클리드 거리
    - 자카르도 거리
  - TF-IDF 기반
    - 코사인 유사도
    - 유클리드 거리

 - 딥러닝 기반
   - AutoEncoder-based Model
   - Softmax DNN
   - GMF
   - MLP-based MF
   - NCF
   
# 연관규칙분석 Association Rules 
"user A는 기저귀를 사면 맥주도 함께 사더라"
user A : 기저귀 -> 맥주


## transaction matrix(sparse)
- rule-based
- 사람이 상호작용(구매, 평가, 시청 등)한 기록을 표로 정리
- pviot table로 변환
  - user가 item을 상호작용했으면 1, 아니면 0
  - sparse matrix 접근법
    1. 아이템 선택 기록이 없으면 0으로 채운다.
    2. 딥러닝으로 sparse matrix을 채워 Dense matrix로 만든다.
  - 행렬 분해 Matrix Factorization
    - size가 큰 sparse matrix를 작은 행렬들로 분해하여 효율적으로 저장 및 처리

## A Priori Algorithm
자주 발생하는 itemset(frequent itemsets)에 대해서 rule을 생성하기
- itemset : 서로 다른 item들의 집합

item의 수가 p개일 때, 가능한 Rule의 수 = 집합의 크기 ^ p - 진 부분집합의 개수


![Apriori](https://www.researchgate.net/publication/337999958/figure/fig1/AS:867641866604545@1583873349676/Formulae-for-support-confidence-and-lift-for-the-association-rule-X-Y.ppm)
![test](https://t1.daumcdn.net/cfile/tistory/273CDE3E573D68DF14)
- Support : 해당 # of Itemset / 전체 # of transaction 의 비율
- Confidence : P(Y|X)의 추정값
  - itemset X 구매 했을 때, item set Y를 구매할 확률
  - Confience 높음 → Strong한 rule (연관관계가 높다)

- Lift : P(Y|X) / P(X)
  - X,Y가 독립일 때 (연관이 없을 때)
  - confidence = support
  - `독립적으로 Y가 일어날 확률`에 비하여 `X가 일어났을 때 Y가 일어날 확률`
    - 1보다 큼 : 연관관계가 높다( 독립적으로 Y 일어날 확률 < X→Y 이므로)
    > `독립적으로 Y가 일어날 확률`의 OOO배다

# 거리 기반
- 다양한 방식으로 벡터간 거리를 구한다.




# reference

- https://rfriend.tistory.com/191
- https://medium.com/analytics-vidhya/association-rule-learning-apriori-algorithm-d4abebbbbbcd

- https://github.com/SeongBeomLEE/Tobigs_Recommendation_System_Seminar/blob/main/Week2_Recommendation_Seminar_Code.ipynb




# 딥러닝 기반

1. tokenizing
2. word2vec (word embedding)
3. 딥러닝 (Autoencoder)


## 현재 많이 사용되는 추천 시스템
1. CNN을 다른 task로 학습 시킴 (어떤 목적으로 imbedded할 지 선택)
- 고양이 / 개를 구분하는 feature
- 낮 / 밤을 구분하는 feature

2. 출력층 바로 직전 layer값을 imbedded value로 사용 (AutoEncoder의 latent dimension와 의미상 비슷)
- 마지막 바로 전층에 Input이 압축되어 포현되어 있을 것이다. = Imbedded 되었다.
- (출력층을 버리면) 특별한 이미지에 대해 Encoding 해주는 Encoder가 된다.
  - 차원이 큰 이미지 -> Embedded 된 값으로 DB에 저장
  - 유사한 이미지 탐색 : Embeeded vector in DB 와 비교
    - 거리 찾기 : 유클리드, 코사인 등등 -> O(N)으로 너무 비용 비쌈
    - Retrieval and Ranking

## Encoder 기반 영상 추천
- Encoder의 분류기 직전의 층 = Input이 압축되어 있을 것이다.
- Imbedding 하는 기능을 함수로 구현한 것과 비슷하다. (학습된 NN을 통과시키면 특정한 coce로 반환됨)
- Data instance를 imbedding 한다.
  - 유사한 의미 (latent dimension / imbbed value)

```
# 이미 학습이 끝난 model의 input과 출력층 직전 layer의 output을 output
encoder = Model(inputs=model.input, outputs=model.layers[-2].output)
```
- encoder의 inputs으로 model NN과 연결됨
- encoder의 ouputs으로 model NN의 직전 layer의 output을 출력함

### DCI KNN
- 빠른 최근접 검색 알고리즘
  - C -> python으로 구현한 코드
  - encoder.pedict(bacth_x) / encoder(bacth_x)

## 샴네트워크를 이용한 추천
  - one-shot learning    
    한 레이블 당 하나의 이미지만 있어도 분류할 수 있게 학습시키는 것

  - [Siamese Neural Networks for One-shot Image Recognition(샴 네트워크)](https://jayhey.github.io/deep%20learning/2018/02/06/saimese_network/)
  
  ![Siamese](https://i.imgur.com/HLQqD3A.png)

  - 2개의 동일한 구조의 CNN을 동시에 학습
  - 같으면 1, 다르면 0으로 학습
  - 같은것과 다른 것을 0.5 비율
  - 

