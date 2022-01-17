# PyTorch


- model.eval() vs with.no_grad()
  - model.eval() : eval mode로 전환
    - 모든 레이어가 eval mode에 들어가도록 해줌
      - 학습할 때만 사용하는 개념인 Droupout / BacthNorm은 비활성
      - training과 inference시 다르게 작동하는 layer 존재!!
    - with.no_grad() : inference mode로 전환
      - 일반적으로 grad를 계산하지 않는 inference 과정에서는 with.no_grad()로 감싸준다.
      - PyTorch의 autograd 기능 비활성 (inference 과정에서는 필요 없으니까)
      - gradient를 트래킹 하지 않는다.(메모리 절약 , 속도 증가)
      - dropout을 비활성화 시키지 않는다.

```
model.eval()
with torch.no_grad():

    for batch in data_loader:
    	#code
```

- optimizer.zero_grad():
  - "Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문"에 우리는 항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작
```
total_loss.backward()
optimizer.step()
optimizer.zero_grad()
```


- model.zero_grad() 와 optimizer.zero_grad() 차이
  - optimizer의 grad만 초기화 해줄 때
    - model 1개 , optimizer 여러 개
  - model의 grad만 초기화 해줄 때 zero_grad()
    - model 여러 개 , optimizer 1 개
    - 등록된 모든 parameter 모두에 대해 tensor.zero_grad() 호출


- optimizer.step()
 
    step()이란 함수를 실행시키면 우리가 미리 선언할 때 지정해 준 model의 파라미터들이 업데이트 된다.


```
loss = ~
# 역전파 단계(backward pass), 파라미터들의 에러에 대한 변화도를 계산하여 누적함
loss.backward() 

# optimizer에게 loss function를 효율적으로 최소화 할 수 있게 파라미터 수정 위탁
optimizer.step() 

# 이번 step에서 쌓아놓은 파라미터들의 변화량을 0으로 초기화하여
# 다음 step에서는 다음 step에서만의 변화량을 구하도록 함 
optimizer.zero_grad() 
```