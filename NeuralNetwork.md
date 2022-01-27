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
    - [[template]pretrained_classification.ipynb](https://github.com/Tieck-IT/Study/blob/main/template/%5Btemplate%5Dpretrained_classification.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/template/%5Btemplate%5Dpretrained_classification.ipynb)

    - [[template]ROCAUC_and_confusionMatrix.ipynb](https://github.com/Tieck-IT/Study/blob/main/template/%5Btemplate%5DROCAUC_and_confusionMatrix.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/template/%5Btemplate%5DROCAUC_and_confusionMatrix.ipynb)
    - [[kaggle]x_ray_chest_kfold.ipynb](https://github.com/Tieck-IT/Study/blob/main/template/%5Bkaggle%5Dx_ray_chest_kfold.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/template/%5Bkaggle%5Dx_ray_chest_kfold.ipynb)



# Layer 구조와 진행 순서
- Input layer -> hidden layer -> Output Layer
    - compile -> fit -> evaluate, predict
- Dropout layer : 일정 비율의 노드를 0으로 만듦 (overfitting 해결)
  - fit(학습) 할 때만 적용됨
  - predict 때는 적용되지 않음
- Dense Layer : 1차원을 입력으로 받는, 퍼셉트론 구조
- Conv2D Layer : Conv 연산을 진행
- (Max/average)Pooling2D Layer
- Bidirectional : 양방향 layer 구조, forward 방향 + Backward 방향
- RepeatVector : 입력을 n번 반복
# compile
## loss : 손실함수 설정
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

## [Optimizer](https://user-images.githubusercontent.com/45377884/91630397-18838100-ea0c-11ea-8f90-515ef74599f1.png)
   
        model.compile(optimizer="RMSprop", ...)
        model.compile(optimizer=RMSprop(learning_rate=0.001), ...)
      
  
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
- 함수를 사용하거나 이미지의 경우 /255.0 으로 가능 (at RGB format)
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



## 교차 검증 _Cross Validation_
      train data set의 크기가 부족할 때 사용
  - 일반적으로 OutOfFold 방식이 사용됨
    - K-fold Cross Validation 등
    - **20%** 의 validation set * **5** = **100%**
      - 서로 다른 5개의 모델로 훈련했기 때문에 앙상블 모델로 여길 수도 있음
  - train dataset의 절대량에 따라 **validation split**의 비율 결정
  - k-fold's split
  - batch_size 는 gpu의 성능에 따라 결정
  

      train set의 데이터 크기에 따라 fold 개수 K를 결정



  일반적으로 test(validation) set의 비율을 20%로 설정하지만 train set이 진짜 작으면 test set을 0.01, 0.05를 쓴다.

  - 오버 샘플링도 데이터 절대량 부족의 한계 극복하기 어려움
    - DL, GAN 역시 데이터 수 적은 한계를 완전히 극복하기 어려움
    - 오버샘플링으로 새로운 데이터를 생성해도 원본과 유사한 데이터가 쌓이는 것이기 때문에 Epoch를 반복하는 효과 
  - 사용시 주의할 점
      - 반복으로 oversampling과 유사한 효과 낼 수 있음
      - Epoch 너무 늘어나면 과적합 우려

  GAN을 이용한 oversampling 아이디어는 많지만 성능이 좋은 알고리즘이 아직 없음


## 성능평가
- [분류 모델 평가 지표](https://github.com/Tieck-IT/proeject/blob/master/README.md)
- plt.scatter(y_test,y_pred)
    - y = x 꼴이면 예측을 잘하는 것
- loss



# functional API 
[이전 층의 출력 = 다음 층의 입력]을 명시하는 표기법

      
  ` input_tensor= Input(inputs = Input(shape=(X_train.shape[1],)))  `

  ` X = Dense(10)(input_tensor)(x) `

  ` X = Dense(10)(x)`


    
# DNN
    - Dense Layer의 단점
        - Input shape : only **1차원**
            - 이미지, 영상 데이터 = 4차원 (색 채널 3차원 + 데이터 갯수 1차원)
            - 3차원 이상인 데이터는 Flatten()으로 1차원으로 변환하는 과정에서, 데이터 손실 발생




# CNN
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
 

 # U-Net
 - 커널 사이즈가 줄어들어 병목 구건(Bottleneck)을 만들고 전반부 Layer와 후반부 Reverse Maxpooling을 통해 증가시킨 Layer를 결합하며 층의 채녈 수를 증가시킨다.
   -  출력 layer로 Conv2d(1,(1,1)) 사용
      - Conv2d가 여러 장의 필터를 1장으로 압축시키는 역할
      - Dense(activation=None) / [GlovalAveragePooling](https://gaussian37.github.io/dl-concept-global_average_pooling/) : 사용 가능
     
 - Input size = Output size
   - [UNET_도로이미지_segmentation.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/UNET_%EB%8F%84%EB%A1%9C%EC%9D%B4%EB%AF%B8%EC%A7%80_segmentation.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/UNET_%EB%8F%84%EB%A1%9C%EC%9D%B4%EB%AF%B8%EC%A7%80_segmentation.ipynb)
    - tf.keras.losses.BinaryCrossentropy(from_logits=True)
      - 내부 구조상 sigmoid 변환 후에 (from_logits= False, Default)로 역 sigmoid로 복원한다.
      - from_logits=True : sigmoid 된 값을 반환하므로 activation fucntion = 'sigmoid' 필요 없음 
    - loss : binary_crossentropy or MSE
        - binary_crossentropy로 이진 분류로 접근 했을 때 val_accuracy의 증가폭이 더 빠른 모습을 보였다.
      - [link](https://utto.tistory.com/8)


# AutoEncoder
  - 입력을 다시 출력으로 사용
    - 입출력 관계를 학습
  - 인코더(DNN) + 디코더(DNN)
  - 큰 차원(input shape) -> 축소된 차원(패턴을 가진 핵심 차원) -> 큰 차원(output shape = input shape)
    - 다른 차원으로 변환 후, 다시 원래 차원으로 되돌아오는 역할
    - 데이터 = 정보(핵심) + 노이즈
      - 데이터에서 노이즈를 제거
      - output이 공간상의 특정 클래스 boundary에 속하면 해당 class로 분류
    - input shape의 공간 상에서 boundary에 class lebel이 붙어있음(ex. MNIST의 0,1,..,8)
    - loss : Accuracy
      - output shape의 공간 class boundary  = input shape의 공간 class boundary인지 체크
    
  - [Reference](https://deepinsight.tistory.com/126)
  - [NoiseRemoval](https://www.youtube.com/watch?v=F7ox6R773OQ)
    - 학습한 패턴의 파형만 제거 가능
    - [[AutoEncoder]denoising_autoencoder.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Ddenoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Ddenoising_autoencoder.ipynb)
  - [M-Net](https://github.com/adigasu/FDPMNet/blob/master/test.py)
    - [[AutoEncoder]mnet_segementation.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dmnet_segementation.ipynb
) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dmnet_segementation.ipynb)
  - [SuperResolution](https://github.com/krasserm/super-resolution)
    - [[AutoEncoder]super_resolution.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dsuper_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dsuper_resolution.ipynb)
    

# Image Segmentation
- [github](https://github.com/divamgupta/image-segmentation-keras)
- [[segementation]unet_segementation_color_image.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5Bsegementation%5Dunet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5Bsegementation%5Dunet_segementation_color_image.ipynb)

# Layer

nn을 구성하는 층


# Pooling
- layer의 크기를 줄일 때 사용

- 목적
  1. input size를 줄임(Down Sampling)
  2. overfitting을 조절

     : input size가 줄어드는 것 =  parameter의 수가 줄어드는 것

  3. 특징을 잘 뽑아냄.

     : pooling을 했을 때, 특정한 모양을 더 잘 인식할 수 있음.
  - [reference](https://supermemi.tistory.com/16)
  
## MaxPooling
  
  `tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None,
    **kwargs)`
- 범용적으로 사용됨
- 정해진 크기 내에서 값이 가장 큰 것을 선택
  - (n,n,3) 에서 (1,1,3) 로 크기 감소 (채널 고정)

## GlovalAveragePooling
  `tf.keras.layers.GlobalAveragePooling2D(
    data_format=None, keepdims=False, **kwargs)`
  - 정해진 크기 내에서 한 채널 값들을 평균

  - 최종 출력에 FC Layer 대신 사용 가능

   - [reference1](https://gaussian37.github.io/dl-concept-global_average_pooling/)
   - [reference2](https://strutive07.github.io/2019/04/21/Global-average-pooling.html)


# RNN
- 시계열 데이터에 적합
- CNN 대비 5 ~ 10배 느림
  - RNN : N개 perceptron이 병렬 처리
  - CNN : 순차적으로 진행
- GRU : LSTM이 아닌 고전적인 방식(일반적인 perceptron 사용) / 성능 : LSTM > GRU
- 딥러닝에서 순차열(시계열) 데이터는 feature 데이터로 변환 후 사용
- Input : 2차원 
- 특정 기간의 데이터로 다음 구간을 예측 
  - 여러개의 데이터를 피처화
    - 예측값 1~6 -> 7(예측)
    - 2~7 -> 8(예측)
    - 기간을 변경하며 반복
- OFFSET : 현재 시점의 바로 뒤
  - OFFSET = 3, 현재시점부터 [t+1,t+2,t+3]을 **출력**하겠다.
  - 마지막 OFFSET의 위치 = 윈도우의 첫번째값 + 윈도우 길이 + OFSSET 수 - 1
  - 시간은 (계절, 월) 등으로 속성으로 취급(순차열로 취급하지 않음. 영향력 낮기 때문)



# AutoEncoder
- Conv2D + AutoEncoder
  - Encoder : 사이즈 감소(max pooling)
  - Decoder : 사이즈 증가 (Upsampling -> Conv2D == Conv2DTranspose)
  - real world : 현재 데이터가 존재하는 공간
    - 측정 가능 (학생의 시험 점수)
  - latent world(잠재 공간) : 이미지의 핵심이 존재하는 공간
    - 직접 측정 불가 (학업 능력)
  
  
  - 전제 : real world에서 군집화 되어 있으면 latent world에서도 군집을 이루고 있을 것
    - Latent world를 지나서 다시 원본 사이즈로 늘리면 핵심이 아닌 것 제거
      - 낙서, 흉터, 금, 이상탐지 등
      - 이상탐지 : 어떤 이미지가 복원 되었을 때 원본하고 차이(픽셀 값 차이, diff)가 크면, 원본과 다른 이미지
    - skip-GANomarly 등에서, AutoEncoder의 input imbedding(encoding된 값) = AutoEncoder의 output imbedding(분류기 직전 층)(encoding된 값)의 차이를 좁히는 것이 트렌드


## 이상 탐지
fraud detection system (fds)
- AutoEncoder 문제
- 비정상의 클래스가 매우 적음 (98% : 2%)
- 학습에 사용되지 않은 데이터는 [real world -> latent world -> real world] 경우 차이가 크다.
  - real world에서의 boundary에서 차이가 크면 이상으로 판별



- 암호화된 열(값 자체가 해싱, 스케일링 등의 과정으로 암호화됨)
- 전통적인 Classification 문제로 안풀림
  - 검증할 때는 2% : 2%, 정상 : 비정상 비율로 맞추고 성능 평가
    - 가급적 오버 샘플링 보다는 언더 샘플링으로(원본 데이터 수정어서 신뢰성 있음)
    - 데이터의 수가 너무 작으면 테스트의 의미가 없음(ex, 5개 : 5개)

- [logit](https://haje01.github.io/2019/11/19/logit.html)

# GAN
- 이상 탐지 문제 
  - D가 정상인 것의 boundary 학습
  - 불균형 라벨에 사용 99%(양품) : 1%(불량품)
- 이미지 번역(변환) _Image translation_
- 아이디어 떠오름 -> mnist로 해보기 -> cifar10으로 해보기
  - 안되면 구현하기 어려운 아이디어
- real world의 데이터 분포는 가우시안 분포 사용

## mode collapse
- 학습 안되기로 유명
  - 우연히 boundary 안에 포함되는 결과를 출력 -> G가 동일한 이미지 생성
- [해결 방법](https://developer-ping9.tistory.com/108)
  - DCGAN
    - Deep Convolutional Generative Adversarial Networks
  - wgan gp
    - 최근 구조, 학습 잘되는 구조
    - wgan에서 gradient 규제
  

## Discriminator
- 판별기 : 이미지 판별 (진짜 / 가짜)
- 위조 지폐 판별가
- 목적 : boundary 학습
  - boundary : 주어진 label과 아닌 것들을 분류


## Generator
- 생성기 : 노이즈에서 가짜 이미지 생성
- 위조 지폐 생성자
- 목적 : boundary안에 포함되는 이미지 생성

- 이미지 = 픽셀, n차원 공간상에 뭉쳐서 구조화 되어 있음
- 공간 상의 같은 군집 = 같은 class



### 학습 과정
1. n차원 공간에 무작위로 생성
2. Discriminator의 boundary(판별 기준) 안으로 들어감
3. Discriminator가 boundary를 좁힘 (더 엄격한 기준)
4. 2 - 3 반복
5. 새로운 데이터가 생성됨


### 기존의 방법의 한계를 극복
- 불균형 라벨 99% : 1% 인 경우


#### boundary 근처에서 샘플링
- 1 (boundary 안의 데이터) : 2^n
  - 고양이 vs 강아지
    - 고양이 100개 샘플링, 강아지 100개 샘플링
  - 고양이 vs 고양이 이외
    - 고양이 100개 샘플링 - 개미, 거미, 강아지, 호랑이 .... 각각 100개씩 샘플링
  - 차원이 늘어날 수록 샘플링 난이도는 매우 증가

## GAN 전이학습
### F 고정 , B 학습
- 전이학습 F를 기존 모델에서 가져옴(고정)
- F의 output Z로 Classifier B를 학습

### F 학습, B 고정


## CycleGan
- 도메인 간 pair가 되는 데이터 찾기
  - 풍경화 - 사진