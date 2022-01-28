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

