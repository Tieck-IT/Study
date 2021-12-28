# Study
study for daily (coding test, linux, DL, ML, stat, DA, tensorflow.teras, etc ... )

- 딥러닝 개발 스택 tensorflow / keras / PyTorch
  - 사용자 코드 : python application
  - 편의성 좋은 프레임 워크 : Keras <- **focus** 
  - 딥러닝 프레임 워크 : Tensorflow / PyTorch <- **focus** 
  - 하드웨어 사용 라이브러리 : Cuda / CuDNN
  - 하드웨어 : CPU / GPU

- trend : tensorflow -> keras -> PyTorch

# my templete for CNN
Start here to Fine Performance (**excute time**, **val_loss**)
  - preprocessing (Data Augmentation)
  - callbacks : 
    - early stopping, ReduceLROnPlatue, RealTimeLossCallback(custom), ModelCheckPoint
  - ImageGenerator
  - Models (Custom / Pretrained)
    - [[template]pretrained_classification.ipynb](https://github.com/Tieck-IT/Study/blob/main/%5Btemplete%5Dpretrained_classification.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/%5Btemplete%5Dpretrained_classification.ipynb)

    - [[template]ROCAUC_and_confusionMatrix.ipynb](https://github.com/Tieck-IT/Study/blob/main/%5Btemplate%5Dauc_and_confusiomMatrix.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/%5Btemplate%5Dauc_and_confusiomMatrix.ipynb)



# Layer 구조와 진행 순서
- Input layer -> hidden layer -> Output Layer
    - compile -> fit -> evaluate, predict
- Dropout layer : 일정 비율의 노드를 0으로 만듦 (overfitting 해결)
  - fit(학습) 할 때만 적용됨
  - predict 때는 적용되지 않음
- Dense Layer : 1차원을 입력으로 받는, 퍼셉트론 구조
- Conv2D Layer : Conv 연산을 진행
- (Max/average)Pooling2D Layer :
# compile
  - loss : 손실함수 설정
    - **feature dataset**
      - 분류
          - categorical_crossentropy : y의 값이 onehot encoding 인 경우 (y is one-hot-encoded)
          - sparse_categorical_crossentropy : onehot encoding을 keras에서 대신 해줌 (y is categorical classes)
          - 출력층 Dense( n, activation func = "softmax")
          - metrics = ["accuracy"]
          - 정말 성능을 높이려면 label 1개를 인식하는 n개의 모델 만들기
      - 이진 분류 (binary)                
          - 1개가 결정되면 다른 라벨의 확률 결정 1 = p + (1 - p)
          - 출력층 Dense(1, activation = "sigmoid")
          - loss = "binary_crossentropy"
          - metrics = ["acc", "AUC", Precision(), Recall()]
      - 회귀
          - 출력층 Dense(1)
          - metrics = ["mse"]
              - mse : [(예측값 - 실제값)^2의 합]의 평균
              - mae : [|예측값 - 실제값|의 합]의 평균
              - mape : mae의 p 백분율(percentage)
        - loss is nan?
          - 스케일이 너무 큰 경우
            - sol 1. 데이터, layer 구조 확인, 재설정
            - sol 2.  kernel_initializer 사용하기 : he_normal( fit for Relu ) / lecun_normal / Xavier Glort(동적 sd 조절)
              - shift parametor를 이용한 느슨한 정규화
              - 좋은 가중치 초기화의 조건
              1. 값이 동일하지 않음
              2. 충분히 작아야 한다.
              3. 적당한 sd를 가져야 한다.
                  - (표준)정규 분포
      - **image dataset**
        분류
          - sparse_categorical_crossentropy (Input label : one hot encoding X)
          - categorical_crossentropy (Input label: one hot encoding O)

          - multi-class (1개의 객체만 존재 + 여러개의 class에 속함)
          - multi-label (1개의 라벨에 속한 여러 객체 존재)
  - [optimizer](https://user-images.githubusercontent.com/45377884/91630397-18838100-ea0c-11ea-8f90-515ef74599f1.png)
   
        model.compile(optimizer="RMSprop", ...)
        model.compile(optimizer=RMSprop(learning_rate=0.001), ...)
      
  

# 학습
- 데이터 총 개수  = batch_size * steps  epochs  * step_size_per_epoch
- test data는 최소 500개 이상 되어야 제대로 학습됨
- Bacth size는 2의 n제곱으로 하는 것이 좋다. (RAM 단위와 동일하게)
- 입력이 0에 가까운 숫자일 수록 학습이 잘된다.

## 전이 학습
- Model의 구성
  - Part A. Feature Extraction Layer
  - Part B. Classification Layer
- 전략 :
  - 성능이 좋다. -> Part A와 Part B가 우수하다. -> Part A의 성능이 좋은 conv_layer 구조 
  -  그대로 사용하고 데이터만 변경하자.

- 미리 훈련된(pretrained) 모델에 Custom dataset을 넣어 추가로 학습시킨다.
- Image dataset은 각 모델의 Input shape에 따라 resize할 수 있다.
  - feature dataset은 # of feature를 맞추기 까다롭다.
    
- [Pretrain model performance list](https://keras.io/api/applications/)
  - [Tensorflow.keras pretrained model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet)
  - EfficientNetB3도 좋음.

# 전처리
- 정규화, 표준화
- input은 숫자 형태 (numpy로 넣기)
    sol 1. df.to_numpy() (type : np.ndarray)
    sol 2. df.astype(float) (type : df)
- activation func을 relu 했을 대 loss가 전부 nan
    - tanh를 쓰니 정상작동
        - target이 너무 커서 그렇다.
        - log 스케일링 한 값의 평균이 5.305 , 6.249

## 정규화
- 함수를 사용하거나 이미지의 경우 /255.0 으로 가능
- 하지 않으면 학습이 더디게 진행되므로 반드시 전처리 과정에 포함

## 배치 정규화 _Bacth Normalization_
- weight의 값이 평균 0 , 분산 1인 상태로 분포
- 이전 Layer의 출력이 다음 Layer의 입력으로 들어가기 전에, 값을 0에 가까운 값으로 만듦
- 학습 단계에서 모든 Feature에 BN을 하면, 동일한 scale이 되어 learning rate 결정에 유리
- Feacture Scale이 다르면, gradient descent가 다르게 되고 같은 learning rate에 대해 weight 마다 반응하는 정도가 달라짐
- 적용 순서 Layer -> **BN** -> activation -> Dropout in DNN
  - 논문 참고하기 + 검증하기
- relu가 먼저 적용되면 음수인 부분이 0이 되어버리므로 BN 적용 전 weight<0인 부분을 잃는다.
  - train set과 test set의 분포가 다르면 Feature 분포가 다르고 Hidden Layer가 깊을 수록 변화가 누적되어 학습이 어려움.
  - 즉, Layer가 깊을 수록 BN의 적용 여부에 따른 차이가 증가해서 기존에 학습했던 분포와 전혀 다른 분포를 갖게 된다.

  - Dropout 되기 전에 써야함
## 규제 _Regularization_
- Layer의 weight 값이 1보다 작은 상태를 유지
- L2 : w^2 < 1
- 사용법
  
  `from tensorflow.keras.regularizers import l1, l2, L1L2`

  `model.add(Dense(20, activation='relu', kernel_regularizer=l2()))`


## Overfitting
- 일반적으로 학습 데이터의 수가 적은 경우에 발생
- 데이터의 수에 비해 모델이 복잡함
- train loss는 감소하지만, val_loss는 일정 epochs 이후부터 감소하지 않는 경향
  - sol 1. 훈련 데이터의 수 늘리기 (데이터 증강 등)
  - sol 2. 모델의 복잡도(파라미터) 줄이기 (Dropout 등)
  - sol 3. 데이터 전처리 (정규화 등)
  - sol 4. 성능(val_loss)이 개선될 때마다 모델을 저장 (model checkpoint)
  - sol 5. 특정 횟수(patience) 동안 성능이 개선되지 않으면 중지 (early stopping)
    

## 성능평가
- plt.scatter(y_test,y_pred)
    - y = x 꼴이면 예측을 잘하는 것
- loss
- metrics

## Callbacks
- Model checkpoint : best_weight 일 때 저장
- Early Stooping : patience 번의 epoch 동안 성능 개선이 없으면 중지
- ReduceLROnPlateau : 성능지표(loss)가 옆으로 횡보하는 고원이 patience 번의 epoch 동안 반복되면 중지
- Custom Callbacks
  - Realtime loss callback fun
  - learning rate scheduler : 일정 기준(epoch 등)로 learning rate 변경


## ImageGenerator
ImageGenerator의 2가지 역할
1. Data Augmentation
2. Bring Data per Batch
- 데이터를 가져와서 전처리하는 일종의 파이프라인
- RAM SIZE <<< DATA SIZE 이기 때문에 전체 데이터 셋을 메모리에 한번에 올리고 처리할 수 없다. 그래서 Batch 단위로 가져와서 처리한다.
  - ImageGenerator 
    - _tf.keras.preprocessing.image.ImageDataGenerator_
    - 
  





    
# dnn
    - Dense Layer의 단점
        - Input shape : only **1차원**
            - 이미지, 영상 데이터 = 4차원 (색 채널 3차원 + 데이터 갯수 1차원)
            - 3차원 이상인 데이터는 Flatten()으로 1차원으로 변환하는 과정에서, 데이터 손실 발생

  - functional API : 이전 층의 출력 = 다음 층의 출력의 흐름을 명시하는 표기법

        
    ` input_tensor= Input(inputs = Input(shape=(X_train.shape[1],)))  `

    ` X = Dense(10)(input_tensor)(x) `

    ` X = Dense(10)(x)`



# cnn
## Convolution (합성곱 연산)
    - 핵심이 되는 Feature를 추출하기 위해 사용
      - Input N * M 의 특징을 추상화, 압축하여 추출 n * m
    - Convolution filter와 동일한 형태의 자극이 왔을 때 값이 최대가 된다.
      - feature map을 추출한다. (입력 값보다 크기가 작아짐)
      - padding으로 가장자리에 0을 추가하여 사이즈 유지 가능
    - Filter (Layer) -> Kernel (Channel)
      - 1개의 필터는 여러장의 Kernel로 구성되어 있다.
    - padding : 가장자리에 0인 데이터를 추가하여 Feature map의 size를 조절하는 방법
      - Feature map 축소에 저항, 원하는 형태의 사이즈로 조절할 때 사용


## Pooling -> Stride
  - Conv 적용된 Feature map의 일정 영역 별로 하나의 값을 추출하여(Max or Average 등) Feature map의 사이즈를 줄임
  - Max pooling : Sharp한 feature(Edge 등) 값을 추출
  - Average pooling : Smooth한 feature 값을 추출
  - GlobalAveragePooling : 채널의 평균으로 값을 추출 (Flatten() 필요 없음 / 차원 축소)
  - 일반적으로 Sharp한 Feature가 Classification에 유리하여 Max pooling이 많이 사용됨
  - 특정 위치의 feature값이 손실 되는 이슈 등으로 최근 CNN에서는 많이 사용되진 않음.

### 단점 보완
  - 최근 CNN에서는 Pooling 대신 Stride를 이용
    - stride : 한 걸음의 크기, 건너뛰는 픽셀의 수


## 저장 포맷
    - .h5 : 하둡에서 사용되는 대용량 포멧 in tersorflow
    - .h5 없으면 tensorflow 포맷
    - 표준 ONNX 포맷
        - TensorRT에서 읽을 수 있음
 
