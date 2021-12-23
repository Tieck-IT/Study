# Study
study for daily (coding test, linux, DL, ML, stat, DA, etc. )




# tf.teras
- 딥러닝 개발 스택 tensorflow / keras / PyTorch
![Untitled](https://user-images.githubusercontent.com/90205987/147028688-0f9bfb09-0311-4738-a471-4ada2a733707.png)
- trend : tensorflow -> keras -> PyTorch

- Input layer -> hidden layer -> Output Layer
    - compile -> fit -> evaluate, predict





- ## compile
    - loss : 손실함수 설정
        - **분류**
            - categorical_crossentropy : y의 값이 onehot encoding 인 경우 (y is one-hot-encoded)
            - sparse_categorical_crossentropy : onehot encoding을 keras에서 대신 해줌 (y is categorical classes)
            - 출력층 Dense( n, activation func = "softmax")
            - metrics = ["accuracy"]
            - 정말 성능을 높이려면 label 1개를 인식하는 n개의 모델 만들기
        - **이진 분류** (binary)                
            - 1개가 결정되면 다른 라벨의 확률 결정 1 = p + (1 - p)
            - 출력층 Dense(1, activation = "sigmoid")
            - loss = "binary_crossentropy"
            - metrics = ["acc", "AUC", Precision(), Recall()]
        - **회귀**
            - 출력층 Dense(1)
            - metrics = ["mse"]
                - mse
                - mae
                - mape : p 백분율(percentage)
    - optimizer
        - "optimizer 계보" 검색
- ## 학습
    - 데이터 총 개수  = batch_size * steps epochs  * step_size_per_epoch
    - test data는 최소 500개 이상 되어야 제대로 학습됨


    ### 배치 정규화 _Bacth Normalization_
      - weight의 값이 평균 0 , 분산 1인 상태로 분포
        - 학습 단계에서 모든 Feature에 BN을 하면, 동일한 scale이 되어 learning rate 결정에 유리
          - Feacture Scale이 다르면, gradient descent가 다르게 되고 같은 learning rate에 대해 weight 마다 반응하는 정도가 달라짐
        - 적용 순서 Layer -> BN -> activation
          - ReLU가 먼저 적용되면 음수인 부분이 0이 되어버리므로 weight의 50%를 날리게 된다.
        - train set과 test set의 분포가 다르면 Feature 분포가 다르고 Hidden Layer가 깊을 수록 변화가 누적되어 학습이 어려움.
          - 즉, Layer가 깊을 수록 BN의 적용 여부에 따른 차이가 증가해서 기존에 학습했던 분포와 전혀 다른 분포를 갖게 된다.
      


  
- ## 전처리
    - logscale 후에 정규화, 표준화
    - input은 숫자 형태 (numpy로 넣기)
        1. df.to_numpy() (type : np.ndarray)
        2. df.astype(float) (type : df)
    - activation func을 relu 했을 대 loss가 전부 nan
        - tanh를 쓰니 정상작동
            - target이 너무 커서 그렇다.
            - log 스케일링 한 값의 평균이 5.305 , 6.249
    
    ```    
    Epoch 8/50
    44/44 [==============================] - 0s 3ms/step - loss: 2715612872704.0000 - mae: 888272.1250 - val_loss: 5482972971008.0000 - val_mae: 1303719.8750
    ```

- ## 성능평가
    - plt.scatter(y_test,y_pred)
        - y = x 꼴이면 예측을 잘하는 것
    - loss
    - metrics
    
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
## Pooling 
  - Conv 적용된 Feature map의 일정 영역 별로 하나의 값을 추출하여(Max or Average 등) Feature map의 사이즈를 줄임
  - stride : 한 걸음의 크기, 건너뛰는 픽셀의 수
  - padding : 가장자리에 0인 데이터를 추가하여 Feature map의 size를 조절하는 방법



## 저장 포맷
    - .h5 : 하둡에서 사용되는 대용량 포멧 in tersorflow
    - .h5 없으면 tensorflow 포맷
    - 표준 ONNX 포맷
        - TensorRT에서 읽을 수 있음
    


# coding test
    - boj
 


