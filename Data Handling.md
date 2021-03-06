# DataHandling
## pandas
- 원핫인코딩
    - pd.get_dummies()
        - columns : 대상 열 or df
        - drop_first = True : n-1개의 열 생성(정보손실 x, 차원축소 o)
    - dummy_na = True : 결측값을 인코딩에 포함

- 전처리 과정에서 test dataset은 **절대로** 수정하면 안된다.
- test dataset에 할 수 있는 유일한 경우는 train 형태와 동일하게 맞추기 위한, train과 동일한 전처리 할 때 뿐이다.
    - reason 1. test dataset은 실제 데이터라고 가정했기 때문
    - 존재하지 않는 미래의 데이터(test dataset)을 어떤 방식으로든 접근 할 수 없다.
    - reason 2. train dataset의 데이터 분포 = 실제 데이터 분포
    - 함께 스케일링 하면 test dataset이 mean, std에 반영되어 데이터 오염.(data leasky)
- Scaling (normalization / stdandization / log / minmax)
## numpy
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
          ~~~
              np.shuffle(x)
              split_index = int(len(X) * train_size)
              X_train, X_train = X[:split_index], X[split_index:]
              y_train, y_train = y[:split_index], y[split_index:]
          ~~~
  - argsort / argmax / argmin : arg가 붙으면 index를 반환
  - vstack / hstack (vertical / horizontal) ([a,b,...])
      - 데이터에 직접 행을 추가 / 삭제/ 삽입 하는 것은 비효율적
        - numpy는 내부적으로 새로 생성되기 때문
      - (v/h)stack은 데이터가 추가(병합)된 것처럼 보여줌
  - a.astype(np.int) : 타입 변경
  - concatenate : 축을 명시적으로 설정 할 수 있다.

    - layer끼리 합쳐서 Block을 만들 때 사용
  
      `X_total = np.vstack([X_train, X_test]) -> np.concatenate([X_train,X_test])`
      
      `y_total = np.hstack([y_train, y_test]) -> np.concatenate([y_train, y_test])`
- expand_dims(a), axis=-1)
  - axis = -1 : 현재 차원 형태를 유지하면서 크기가 1인 차원을 생성
- squeeze(a, axis= None) : 크기가 1인 차원을 제거

- 축 k를 없애는 차원 축소 연산자
    - sum(data, axis=k)
    - max(data, axis=k)
    - next(data, axis=k)
    - etc ...
    - axis = (a, b, ... ) / tuple 전달 가능

- eye : 항등행렬 I 생성
- onehot encoding
  
  `category_index * np.eye(n)`

  `np.eye(n) [category_index]`

  - decoding
      `np.argmax(ohe_matrix)`

- 팁
    - shape의 첫번째는 데이터 갯수, shape[1:n]의 형태인 데이터가 shape[0]만큼 존재
    - onehot encoding : np.eye
        - recovering to categorical : np.argmax
    - numpy.array가 tensorflow 내부에서 tensor로 전환된다.
      - 이미지 로드
        1. 읽기 -> 리스트에 저장
            
            labels.append(plt.imread(file_name))

            labels = np.expand_dims(labels, axis=-1)
        2. 이미지가 저장된 리스트를 np.array로 변환
            - 참고 : [kaggle]kaggle_brainMRI.ipynb
            
          
# pip
 - !pip uninstall -y keras
     -  -y : yes 메시지 전달(중단 없이 실행 가능)


# linux 기본 명령어
  - % vs ! 
    - ! : 파이썬 코드가 아니라 리눅스 코드임을 명시
    - % : 실행 후에도 그 영향력을 지속시킴
  - ls (-option) (DIR): list 
      - a : all, 모든 파일 보기(숨긴파일 까지) (축약형 la)
      - l : long, 세로로 길게 형태로 보기 (축약형 ll)
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
      - d : <파일명> -d <압축해제 dir>
  - tar xvfz (파일명):
      - linux의 기본 압축 해제 명령어

  - 명령어 A |(파이프) 명령어 B
      - 명령어 a의 결과를 바탕으로 명령어 B 실행
  - wget (url): web get
      - web에서 url을 다운
      - O : 파일명 지정
  - curl -L -O <github 다운로드 링크>
    - http 메시지를 쉘상에서 요청하여 결과를 확인하는 명령어 이며, curl 명령어는 http를 이용하여 경로의 데이터를 가져온다
    - O : 서버 파일 이름 변경 없이 다운로드
    - L : url이 가르키는 redirect URL까지 접속함
    - 응용 : 깃허브에서 단일 파일 다운받기 from [link](https://dreamlog.tistory.com/611)
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

# 이미지
- padding : 0으로 채움
- resize : 적절한 값을 계산
  - 처리시간 : padding << resize
    - skimage.transform.resize(image, (256,256)) << cv2.resize(image, (256,256),interpolation=cv2.INTER_AREA)
    - cv2의 처리 속도가 더 빠르다.
- images_list = glob.glob("data/obj/Raccon/train/*[jpg|png|jpeg]")
- [구글튜토리얼](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=ko)
- [텐서플로우허브](https://www.tensorflow.org/hub) : tf로 구성된 다양한 모델 공개


## 이미지 라벨링
- [makesense.ai](https://www.makesense.ai/)
- [Roboflow](https://app.roboflow.com/mg-park)


# train/test split for DataGenerator
- 수동 / 자동 으로 나누기
- [Data_Handing_manually.ipynb](https://github.com/Tieck-IT/Study/blob/main/template/Data_Handing_manually.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tieck-IT/Study/blob/main/template/Data_Handing_manually.ipynb)
