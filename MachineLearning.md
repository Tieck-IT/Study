# Machine learning
학습은 Training data set의 분포 하에서 진행되어야 한다.

Validation data set과 Test data set은 실제 상황을 가정했기 떄문에, 다룰 때 주의해야한다.

- Training data set으로 훈련 : 
자유롭게 전처리, 수정 가능 for performance

- Validation set으로 훈련 과정 중에 성능 평가 : Training data set과 동일한 전처리

- Test data set으로 실제 상황 가정 예측 : Training data set과 동일한 전처리


- 실제 상황의 데이터 = _아직 존재하지 않는 미래_ 의 데이터

- 특히 스케일링(표준화, 정규화, 최대최소)를 할 때 data set을 합쳐서 처리하면 평균, 표준편차에 데이터가 섞여 들어가서 데이터 누수가 발생한다. (Data Leakage)
  - 이 밖에 라벨 인코딩 할 때 역시 Traning data set에 존재하지 않는 Validation, Test set의 데이터는 


# 결측치를 채우는 방법
- 의미를 해석하여 의미상 결측치 처리
    - ex. 해당 없음 99999, 비어있음 -1 등
- 첫번째 시도의 경우 : 평균값,중앙값(숫자형), 최빈값(범주형)으로 채움 (결측치의 비율이 높을 경우 결과에 상당한 영향을 미침)(주의!)
    - 어떤 영향을 끼치는지 면밀히 살펴봐야함.
- 정규분포: 해당 열의 mean, sd를 이용 (MCAR, MAR의 경우)
- 제거 : 결측치의 비율이 상당히 높을 경우 (20% ~ 25%)

_from [결측값 결측치 종류 및 대체](https://seeyapangpang.tistory.com/9)_

## 범주형 변수
1. Target Mean, Frequency
    - Target의 class 별 경향성을 반영

## 숫자형 변수
1. 








***
모델별 특징

# Catboost
- 결측치 x
- categorical 변수 처리에 뛰어난 성능
- categorical feature input    
    - 라벨 인코딩된 feature의 타입을 category 변경
    - 라벨 안된 문자형 feature의 타입을 ㅡcategory 변경
    

# xgb
- 결측치 o
- 숫자형 입력
    - 원핫
    - 라벨
    - 바이너리
    - 타겟민, catboost  encoder
- gpu hist,  글카 사용 가능 ㅡ 매우 빠름
    - 캐글 동작 확인
    

# lgbm

- 결측치 o
- 숫자형  입력
    - xgbm 다음세대, 빠르고 가벼움

