
    
# DNN
    - Dense Layer의 단점
        - Input shape : only **1차원**
            - 이미지, 영상 데이터 = 4차원 (색 채널 3차원 + 데이터 갯수 1차원)
            - 3차원 이상인 데이터는 Flatten()으로 1차원으로 변환하는 과정에서, 데이터 손실 발생




# CNN
## Convolution (합성곱 연산)
    핵심이 되는 Feature를 추출하기 위해 사용
    → Input N * M 의 특징을 추상화, 압축하여 추출 n * m
    Convolution filter와 동일한 형태의 자극이 왔을 때 값이 최대가 된다.
    → feature map을 추출한다. (입력 값보다 크기가 작아짐)
    → padding으로 가장자리에 0을 추가하여 사이즈 유지 가능
    Filter (Layer) -> Kernel (Channel)
    → 1개의 필터는 여러장의 Kernel로 구성되어 있다.
    padding : 가장자리에 0인 데이터를 추가하여 Feature map의 size를 조절하는 방법
    → Feature map 축소에 저항, 원하는 형태의 사이즈로 조절할 때 사용


## Pooling -> Stride
    - Conv 연산이 적용된 Feature map의 일정 영역 별로 하나의 값을 추출하여(Max or Average 등) Feature map의 사이즈를 줄임
    - Max pooling : Sharp한 feature(Edge 등) 값을 추출
    - Average pooling : Smooth한 feature 값을 추출
    - GlobalAveragePooling : 채널의 평균으로 값을 추출 (Flatten() 필요 없음 / 차원 축소)
    - 일반적으로 Sharp한 Feature가 Classification에 유리하여 Max pooling이 많이 사용됨
    - 특정 위치의 feature값이 손실 되는 이슈 등으로 최근 CNN에서는 많이 사용되진 않음.

### 단점 보완
  - 최근 CNN에서는 Pooling 대신 Stride를 이용
    - stride : 한 걸음의 크기, 건너뛰는 픽셀의 수





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
  ```python
  tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None,**kwargs)
  ```
  
- 범용적으로 사용됨
- 정해진 크기 내에서 값이 가장 큰 것을 선택
  - (n,n,3) 에서 (1,1,3) 로 크기 감소 (채널 고정)

## GlovalAveragePooling
  ```python 
  tf.keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False, **kwargs) 
```
  - 정해진 크기 내에서 한 채널 값들을 평균

  - 최종 출력에 FC Layer 대신 사용 가능

   - [reference1](https://gaussian37.github.io/dl-concept-global_average_pooling/)
   - [reference2](https://strutive07.github.io/2019/04/21/Global-average-pooling.html)
   - [reference3](https://gaussian37.github.io/dl-concept-global_average_pooling/)

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
>다른 차원으로 변환 후, 다시 원래 차원으로 되돌아오는 역할

큰 차원(real world)   
🠖 축소된 차원(latent world, 패턴을 가진 핵심 차원)  
🠖 큰 차원(output shape = input shape)  


- Conv2D/DNN 등으로 구성
- Encoder + Bottleneck + Decoder
- Encoder : 사이즈 감소(max pooling)
- Decoder : 사이즈 증가 (Upsampling -> Conv2D == Conv2DTranspose)
- real world : 현재 데이터가 존재하는 공간
  - 측정 가능 (학생의 시험 점수)
- latent world(잠재 공간) : 이미지의 핵심이 존재하는 공간
  - 직접 측정 불가 (학업 능력)

## 원리
>인코더에서 feature map을 추출하며 차원을 감소시키고, 디코더에서 Upsampling(ex, unpooling / Conv2DTranspose[= deconvolution])하여 input과 output의 크기가 동일하다.

- unpooling : maxpooling 할때의 Index를 기록해두었다가, 해당 Index 를 기반으로 Pooling을 역으로 수행
- Transposed Convolution : 일반적인 Convolution 연산의 Transpose
    입력을 다시 출력으로 사용

      입출력 관계를 학습
      

      데이터 = 정보(핵심) + 노이즈
       - 데이터에서 노이즈를 제거
       - output이 공간상의 특정 클래스 boundary에 속하면 해당 class로 분류
      input shape의 공간 상에서 boundary에 class lebel이 붙어있음(ex. MNIST의 0,1,..,8)
      loss : Accuracy
       - output shape의 공간 class boundary  = input shape의 공간 class boundary인지 체크

## 전제
> real world에서 같은 클래스라면 latent world에서 군집을 이루고 있을 것

    Latent world를 지나서 다시 원본 사이즈로 늘리면 핵심이 아닌 것 제거
      - 낙서, 흉터, 금, 이상탐지 등
      - 이상탐지 : 어떤 이미지가 복원 되었을 때 원본하고 차이(픽셀 값 차이, diff)가 크면, 원본과 다른 이미지
      - skip-GANomarly 등에서, AutoEncoder의 input imbedding(encoding된 값) = AutoEncoder의 output imbedding(분류기 직전 층)(encoding된 값)의 차이를 좁히는 것이 트렌드





_from [UPSAMPLING: UNPOOLING, DECONVOLUTION, TRANSPOSED CONVOLUTION](https://hugrypiggykim.com/2019/09/29/upsampling-unpooling-deconvolution-transposed-convolution/)_
    
  - [Reference](https://deepinsight.tistory.com/126)
  - [NoiseRemoval](https://www.youtube.com/watch?v=F7ox6R773OQ)
    - 학습한 패턴의 파형만 제거 가능
    - [[AutoEncoder]denoising_autoencoder.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Ddenoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Ddenoising_autoencoder.ipynb)
  - [M-Net](https://github.com/adigasu/FDPMNet/blob/master/test.py)
    - [[AutoEncoder]mnet_segementation.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dmnet_segementation.ipynb
) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dmnet_segementation.ipynb)  


  - [SuperResolution](https://github.com/krasserm/super-resolution)  
    - [[AutoEncoder]super_resolution.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dsuper_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5BAutoEncoder%5Dsuper_resolution.ipynb)  
  

## 이상 탐지 _Anomaly Detection_
fraud detection system (fds)
***
    AutoEncoder 문제
    비정상의 클래스가 매우 적음 (98% : 2%)
    학습에 사용되지 않은 데이터는 [real world -> latent world -> real world] 경우 차이가 크다.
    - real world에서의 boundary에서 차이가 크면 이상으로 판별
    암호화된 열(값 자체가 해싱, 스케일링 등의 과정으로 암호화됨)
    전통적인 Classification 문제로 안풀림
    - 검증할 때는 2% : 2%, 정상 : 비정상 비율로 맞추고 성능 평가
      - 가급적 오버 샘플링 보다는 언더 샘플링으로(원본 데이터 수정어서 신뢰성 있음)
      - 데이터의 수가 너무 작으면 테스트의 의미가 없음(ex, 5개 : 5개)

- [logit](https://haje01.github.io/2019/11/19/logit.html)


## U-Net
***
커널 사이즈가 줄어들어 병목 구건(Bottleneck)을 만들고 전반부 Layer와 후반부 Reverse Maxpooling을 통해 증가시킨 Layer를 결합하며 층의 채녈 수를 증가시킨다.


  -  출력 layer로 Conv2d(1,(1,1)) 사용
      - Conv2d가 여러 장의 필터를 1장으로 압축시키는 역할
      - Dense(activation=None) / GlovalAveragePooling : 사용 가능
     
 - Input size = Output size
    - tf.keras.losses.BinaryCrossentropy(from_logits=True)
      - 내부 구조상 sigmoid 변환 후에 (from_logits= False, Default)로 역 sigmoid로 복원한다.
      - from_logits=True : sigmoid 된 값을 반환하므로 activation fucntion = 'sigmoid' 필요 없음 
    
    
    - loss : binary_crossentropy or MSE
        - binary_crossentropy로 이진 분류로 접근 했을 때 val_accuracy의 증가폭이 더 빠른 모습을 보였다.  
      _from [here](https://utto.tistory.com/8)_

   - [UNET_도로이미지_segmentation.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/UNET_%EB%8F%84%EB%A1%9C%EC%9D%B4%EB%AF%B8%EC%A7%80_segmentation.ipynb)
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/UNET_%EB%8F%84%EB%A1%9C%EC%9D%B4%EB%AF%B8%EC%A7%80_segmentation.ipynb)
## Image Segmentation
- [github](https://github.com/divamgupta/image-segmentation-keras)
- [[segementation]unet_segementation_color_image.ipynb](https://github.com/Tieck-IT/Study/blob/main/tf_keras/%5Bsegementation%5Dunet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/tf_keras/%5Bsegementation%5Dunet_segementation_color_image.ipynb)





## GAN
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