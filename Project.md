<proeject list>

# 분류 모델 평가 지표  
  
  ## 혼동 행렬 (Consucion Matrix)
  
  ![image](https://user-images.githubusercontent.com/90205987/147517541-06115069-a112-4fd1-8a98-b27650e6bd51.png)

  
    [출처 : https://sumniya.tistory.com/26 ] 
  
  
  
  - 정확도 : 옳게 예측한 비율
  
  ![image](https://user-images.githubusercontent.com/90205987/147515859-d907a5b9-3895-481c-aebd-1974b02204e5.png)

  
  - 정밀도 : 실제 중 [(예측 : 양성) -> (실제 : 양성) ] 예측 비율
    - 범죄자(P)라고 예측한 비율 중 실제 범죄자(P)의 비율

  
  ![image](https://user-images.githubusercontent.com/90205987/147515826-a0cef040-97c9-4fd0-be46-3df552f077a2.png)

  
  - 재현율 : 실제 양성(Positive) 중  [(예측 : 양성) -> (실제 : 양성) ] 예측 비율
    - 실제 암환자(P) 중 찾아낸 비율
  
  ![image](https://user-images.githubusercontent.com/90205987/147515828-e5b5216c-79de-4830-b2fe-a6aeb9358721.png)
 
  - F1 score : 정밀도와 재현율의 조화평균 (불균형 라벨의 평가 보완)
  
  ![image](https://user-images.githubusercontent.com/90205987/147515835-439fc4a7-075d-44f8-8477-ce124718e9d5.png)

  [Select Precision vs Recall](https://sumniya.tistory.com/26) /
  [1종오류 vs 2종오류](https://blog.naver.com/PostView.naver?blogId=parksehoon1971&logNo=221611771475&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView)
   
  
  - [Calculate Average Option of [ Accuracy / Precision / Recall ]](https://rython.tistory.com/14) in multi-class
    - None : default=’binary’ (positive label을 기준으로 계산)
    - Micro Average : 클래스 i의  total [TP / FN / FP/ TN]을 이용해 계산
    - Macro Average : 각 클래스의 [precision / recall / f1 score ]을 구하고 합쳐서 최종 [precision / recall / f1 score ]을 도출
    - Weighted Average : 각 클래스의 data 개수 가중치를 둔, 가중 평균 Macro Average
    
  
  - 손실  그래프
  - 정확도 그래프
  - [AUCROC](https://bioinformaticsandme.tistory.com/328)
    - Positive 결정 임계치에 따른 성능 그래프
    - AUC : 0.5 일 때 최악, Positive / Negative 를 전혀 구분 못함
    - AUC : 1.0 일 때 최상, Positive / Negative 를 완벽히 구분함
    - x축 : FPR (= 1 - precision)
    - y축 : TPR  (= recall)
   
  
  

# [당뇨병 위험 예측 모델](https://github.com/Tieck-IT/proeject/tree/master/PredictDiabetes)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/proeject/blob/master/PredictDiabetes/SimpleModel.ipynb)
  
  
  
  - 임상검사와 방문없이, 설문조사 데이터를 이용해 예측 모델 훈련
  - 사용 데이터 : 교육용 데이터 from [질병관리청 국립보건연구원](https://www.kdca.go.kr/contents.es?mid=a40504070100)
    - 라벨 비율
      - 0  :  0.92925 / 1  :  0.07075
    - Accuracy : 0.8875
    - AUCROC : 0.7831
    - F1 score : 0.2268
    - Recall : 0.2340
    - Precision : 0.22
  
      ![image](https://user-images.githubusercontent.com/90205987/147515793-c54ecf7d-d6cb-4aaa-a669-0d03f9b62411.png)

  
# [시각장애인을 위한 옷 스타일링 시스템](https://github.com/Tieck-IT/proeject/blob/master/ClothStyling4Blind/MyStyling4Blind.ipynb)
  - yolov3-tiny를 이용하여 전이학습
  - 데이터 : 미국 코넬대학교의 [arxiv](https://github.com/switchablenorms/DeepFashion2)에서 공개한 DeepFashion2 Dataset
    - 87.3만개의 데이터셋 중 3.4만개의 검증 데이터셋을 학습에 사용

# [리디북스 도서추천 시스템](https://github.com/Tieck-IT/proeject/blob/master/Ridi_recommender_system/content_system.ipynb)
  - 코사인 유사도 기반 유저 기반 추천 / 도서 기반 추천 모델 훈련
  - sklearn-Surprise 라이브러리의 SVD(잠재요인 행렬분할) 활용
  - 데이터 획득 경로 : 크롤링 (코드 비공개)


  # [정상 / 초기 / 말기 녹내장 탐지](https://github.com/Tieck-IT/proeject/blob/master/paper/glaucoma_dectetion.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/proeject/blob/master/paper/glaucoma_dectetion.ipynb)
 
 - result
  
![image](https://user-images.githubusercontent.com/90205987/149133038-e2f8e8a4-fcd5-4a59-badc-4ce4f70c4af7.png)
![image](https://user-images.githubusercontent.com/90205987/149133070-691b99cb-64ca-4a07-9f05-73f752c64a80.png)
![image](https://user-images.githubusercontent.com/90205987/149133077-89d4f398-3f17-4020-b573-5945c22093ea.png)

  
  ![image](https://user-images.githubusercontent.com/90205987/149133205-72b9f409-71f9-4019-8ea6-27f9d03e4465.png)

  
  - data : 3 class, Retinal fundus images (normal / early / advanced)
    - from : Machine learn for glaucoma, HAVARD Dataverse [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC)
  - how : 
  - reference : Development and Validation of Deep-Learning Algorithm for Electrocardiography-Based Heart Failure Identification
Joon-myoung Kwon
- paper : A deep learning model for the detection of both advanced and early glaucoma using fundus photography , Jin Mo Ahn
  - link is [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207982)
- source : HARVARD Dataverse - Machine learn for glaucoma
  - link is [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC)

