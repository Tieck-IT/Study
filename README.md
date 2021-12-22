# Study
study for daily (coding test, linux, DL, ML, stat, DA, etc. )


# Data Handling
- pandas


- numpy
    - random
        - randint(n,size) (a, b, size)
            - 0(a) ~ n(b)까지의 임의의 정수 size 개 반환
        - rand(n) (n, m)
            - 0 ~ 1의 균등 분포 uniform array(n) , matrix(n,m) 반환
        - randn(n, m)
            - 평균 0, 표준편차 1의 표준정규분포 난수 array(n) matrix(n,m) 반환
        - normal(mu, sigma**2, size)
            - sigma * np.random.randn(...) + mu와 동일
        - shuffle(data)
            - data의 순서를 섞는다. (반환 X)
                - split_index = int(len(X) * train_size)
                - X_train, X_train = X[:split_index], X[split_index:]
                - y_train, y_train = y[:split_index], y[split_index:]
    - argsort / argmax / argmin : arg가 붙으면 index를 반환
    - vstack / hstack (vertical / horizontal) ([a,b,...])
        - a라는 데이터에 행을 추가 / 삭제/ 삽입 하는 것은 비효율적
        - (v/h)stack은 데이터가 추가(병합)된 것처럼 보여줌
    - a.astype(np.int) : 타입 변경

    - 축 k를 없애는 차원 축소 연산자
        - sum(data, axis=k)
        - max(data, axis=k)
        - next(data, axis=k)
        - etc ...
        - axis = (a, b, ... ) / tuple 전달 가능

    - eye : 항등행렬 I 생성
        - onehoten_coded
            - category_index * np.eye(n)
            - np.eye(n) [category_index]


    - 팁
        - shape의 첫번째는 데이터 갯수, shape[1:n]의 형태인 데이터가 shape[0]만큼 존재
        - onehot encoding : np.eye
            - recovering to categorical : np,argmax
          






# linux 기본 명령어
    - ls (-option) (DIR): list 
        - a : 모든 파일 보기(숨긴파일 까지) (축약형 la)
        - l : 세로 리스트 형태로 보기 (축약형 ll)
        - ls -al 형태가 많이 사용됨 
    - cp (-option) (복사 대상 DIR) (복사 후 DIR): copy
        - r : recursive (하위 항목까지 포함)
    - rm (제거 대상 DIR): remove
        - f : force (강제로, Y / N 질문 생략)
        - r : recursive
    - mv (대상 DIR) (이동 DIR): move
        - 파일 이동 기능
        - 이름 변경 기능
    - mkdir (-Option) (생성할 DIR): make directory
        - p : parent, 부모 dir 까지 생성하라
        - mkdir - p sub1/sub2/sub3
    - apt :
        - 명령어가 모여있는 저장소
        - apt-get update : apt 버전 업데이트
        - apt install (설치 대상명)
            - apt install tree
    - tree :
        - 현재 디렉토리를 기준으로 디렉토리 구조를 tree로 보여줌
        -d : 디렉토리 구조만 보기
        - tree > 파일명.txt : 출력 저장 (>기호는 대부분의 명령어에서 결과 저장 기능 수행)
    - zip / unzip (파일명):
        - 압축 / 압축 해제
        - 설치 안된 서버도 존재
    - tar xvfz (파일명):
        - linux의 기본 압축 해제 명령어

    - 명령어 A |(파이프) 명령어 B
        - 명령어 a의 결과를 바탕으로 명령어 B 실행
    - wget (url): web get
        - web에서 url을 다운


# tf.teras
- 딥러닝 개발 스택 tensorflow / keras / PyTorch
    ![Untitled](https://user-images.githubusercontent.com/90205987/147028688-0f9bfb09-0311-4738-a471-4ada2a733707.png)
    - trend : tensorflow -> keras -> PyTorch
    - Input layer -> hidden layer -> Output Layer
        - compile -> fit -> evaluate, predict

    - compile
        - loss : 손실함수 설정
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
                    - mse
                    - mae
                    - mape : p 백분율(percentage)
        - optimizer
            - "optimizer 계보" 검색
    - dnn
        - Dense Layer의 단점
            - Input shape = 1차원
                - 이미지, 영상 데이터 = 4차원 (색 채널 3차원 + 데이터 갯수 1차원)
                - 3차원 이상인 데이터는 Flatten()으로 1차원으로 변환하는 과정에서, 데이터 손실 발생
    - cnn
        - functional
        - convolution (합성곱 연산)
    - 저장 포맷
        - .h5 : 하둡에서 사용되는 대용량 포멧 in tersorflow
        - .h5 없으면 tensorflow 포맷
        - 표준 ONNX 포맷
            - TensorRT에서 읽을 수 있음

# coding test
    - boj
 

 # git
    - add (파일명/ .[all])
        - stage에 데이터를 추가한다.
    - status
        - stage에 추가된, changed 리스트 확인
    - commit
        - repository로 전달할 데이터 선정(준비)
        - m "text": message, 커밋에 대한 간략한 내용
    - push
        - commit 된 파일을 repository에 반영
    - pull request
        - fork 하거나 협업한 repository를 원본에 반영하고 싶을 때, 당겨(pull) 달라고 요청(request)
    - init
        - .git 숨김 폴더 생성
        - 해당 디렉토리를 local로 설정 (환경 설정)
    - config --list
        - user.name, user.email 등을 확인
    - 팁
        - local dir 구조와 git의 dir 구조를 동일하게 유지하자.
        - .git이 존재하는 하위 dir에 .git을 생성하면 꼬인다.
        - .git은 유일하다. (하위 dir까지 포함)
        - tree (> 파일명.txt)로 dir 구조를 저장, 확인 할 수 있다.
