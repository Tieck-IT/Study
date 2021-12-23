# Data Handling
- ## pandas
    - 원핫인코딩
        -pd.get_dummies()
            - columns : 대상 열 or df
            - drop_first = True : n-1개의 열 생성(정보손실 x, 차원축소 o)
            - dummy_na = True : 결측값을 인코딩에 포함



- ## numpy
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
        - onehot encoding
          - 
            `category_index * np.eye(n)`

            `np.eye(n) [category_index]`


    - 팁
        - shape의 첫번째는 데이터 갯수, shape[1:n]의 형태인 데이터가 shape[0]만큼 존재
        - onehot encoding : np.eye
            - recovering to categorical : np.argmax
          






# linux 기본 명령어
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
    - tar xvfz (파일명):
        - linux의 기본 압축 해제 명령어

    - 명령어 A |(파이프) 명령어 B
        - 명령어 a의 결과를 바탕으로 명령어 B 실행
    - wget (url): web get
        - web에서 url을 다운
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