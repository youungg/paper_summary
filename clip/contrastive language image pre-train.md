# clip : contrastive language image pre-train

clip 목적 : 어떤 다운스트림 태스크의 특정하게 튜닝되어 있는 모델이 아닌 뭔가 일반적인 태스크를 수행할 수 있는 이미지 비전 태스크를 수행할 수 있는 모델을 훈련한 것

얘네를 훈련을 위해 이미지넷과 같은 데이터셋이 아닌 WiT라는 웹크롤드 데이터셋을 사용

(크롤링x :  쿼리를 날려 모은 데이터셋 이기 때문)

위 데이터 셋을 가지고 기존 다양한 테스크들의 제로샷으로(fine tuning 없이) inference를 해서 성능을 측정해 보니 생각보다 꽤 잘하더라. 리니어 프로빙을 한 것만큼 잘하더라

zero shot : tuning 없이 downstream task 평가를 진행하였다 + 다운스트림 테스트에 사용된 그 데이터셋 쓰지 않고 wit로 학습을 진행하였다

# Background

- Image classification
    - 한계점
        - 데이터셋의 generality가 떨어짐 (다양한 task의 representation을 반영하지 못함)
        - 새로운(unseen) 데이터 학습이 필요한 경우 → 영상 촬영, 레이블링 등 관련 비용 발생
- 본 논문에서는 zero shot 성능을 보장하는 일반화된 사전 학습 모델을 만들고자 함

- **clip : connection text and images**
- → 웹 기반 image -text pair 데이터 셋으로 contrastive learning 수행

---

비전 모델을 학습시키고 싶은데 거기에는 일반적으로 굉장히 많은 텍스트 캡션 혹은 디스크립션이 달려 있으니 그 텍스트로부터 뭔가 슈퍼비전을 얻어 가지고이 이미지 클래식케이션을 하는 모델을 좀 더 잘 훈련시킬 수 있지 않을까라는 motivation

---

# Approach

- Natural language supervision
    - 이미지와 이미지를 설명하는 자연어를  pair로 학습
    - 언어의 의미론적 representation 학습
    - 자연어를 label로 사용함으로써 인터넷에 존재하는 방대한 데이터를 추가비용없이 사용가능
- sufficiently large dataset
    - 인터넷으로부터 추출된 약 4억개의 이미지-텍스트 pair 데이터(webimag text)
    - 텍스트 query는 위키피디아에서 100번 이상 등장하는 단어 50만개로 구축
    - 데이터 균형을 위해 query 당 이미지 텍스트 pair개수를 최대 2만개로 제한
    - Imagenet의 약 30배 이상 규모
- 

![Untitled](contrastive%20language%20image%20pre-train/Untitled.png)

- 

![Untitled](contrastive%20language%20image%20pre-train/Untitled%201.png)

- 텍스트 인코더 거쳐 가지고 들어가 →, 토크나이제이션 → 트랜스포머 쓰는 경우에는 포지션 임베딩도 같이 들어감 → 그렇게 해가지고 나온 이제 CLS벡터 같은 거를 이제 사용→텍스트임베딩
- image encoder도  Resnet50 or VIT 사용
- positive/ negative로 진행
- positive는 증가 시키는 방향/ negative는 감소 시키는 방향
- normaliztion 진행한것이 차이점

![Untitled](contrastive%20language%20image%20pre-train/Untitled%202.png)

![Untitled](contrastive%20language%20image%20pre-train/Untitled%203.png)

![Untitled](contrastive%20language%20image%20pre-train/Untitled%204.png)

- prompt engineering : 워드 단위의 레이블들에 대해서 뭔가 좀 더 문장 형식으로 만들어주는 과정

![Untitled](contrastive%20language%20image%20pre-train/Untitled%205.png)

- similarity 계산후  softmax 해주고 top5 뽑기
- 위 내용중 중요 내용은 prompt engineering  : 얘네를 얼마나 잘하는냐에 따라 image classification 성능이 천차만별
- image encoder는 다양한  이미지 디스크립션으로 퍼스티브 페어링이 되어서 훈련이 되어있다 →이미지랑은 조금 더 잘 매칭이 되기 위해서는 그 클래스의 레이블 텍스트 자체가 조금 더 문장 형식이어야 된다 →근데 이제 보통 그 뭐 이미지넷 같은 classification 매지막 보면은 클래스에 이름 자체가 단어잖아요 그렇기 때문에이 단어를 조금 더 풀어서 조금 더 디스크립션 형태와 유사하게 만들어주는 역할을 하는게 프롬프트 엔지니어링
- 거꾸로 얘기하면 이제 만약에 클립을 사용해도 prompt engineering 잘 못하면 성능이 그렇게 좋지 않습니다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%206.png)

- batch 사이즈가 크다 보니까 이게 그 메모리를 많이 쓰게 된다 → 그래서 저런 엔지니어링 기법을 동반하지 않을 수가 없을 것 같다
- 추가로 optimize 기법 : grad cache

![Untitled](contrastive%20language%20image%20pre-train/Untitled%207.png)

- 

![Untitled](contrastive%20language%20image%20pre-train/Untitled%208.png)

- prompt로 텍스트 인베딩 벡터들을 계산을 하면은 그것에 대해서 이제 인베딩 스페이스에서 앙상블을 계산을 한다 →임베딩 스페이스에서 피쳐들을 에버리지를 하고 classification 를 한번 돌리도록 작업

![Untitled](contrastive%20language%20image%20pre-train/Untitled%209.png)

- 27가지 데이터셋이 주로 언급
- 12가지 데이터셋은 기존의 이미지넷 모델이 트랜스퍼가 이제 잘 되는지를 연구된 데이터
- 추가적으로 다른 다운스트림 태스크의 데이터셋 15가지
- 12가지 데이터셋은 대체적으로 제너럴한 오브젝트를 포함하는 데이터셋
- 15가지 데이터셋은 더 복잡한 좀 더 이제 특화된 그런 타운 스트림 task 관련 데이터

우측 그림 통해서

→ 복잡하고 특화된 데이터는 zero shot clip 성능이 낮고

→ 일반화된 data는 zero shot clip 성능이 높다

→우측 그림 위를 통해 쉬운  task는 튜닝을 한 것보다 안한것이 성능이 더 좋다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2010.png)

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2011.png)

→ 그래프에서 원샷과 투샷의 진행한 clip 모델보다 성능이 더 좋은 것이 zero shot clip이 성능이 더 좋다

→어설프게 few shot을 주면 generality를 해치기 때문에 성능이 안좋다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2012.png)

- few shot 진행 시, zero shot 성능을 매칭하려면 얼마만큼의 label 데이터가 필요한가

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2013.png)

- gray dot line은 zero shot과 linear probe clip 성능이 같은 지점
- 그래프상 아래로 치우쳤다 → linear probe clip 성능이 더 좋다
- → 위에 있는 애들은 linear 성능도 좋지만 zero shot 성능도 좋다

→ 선형 probe 성능이 높다 : 일반적인 태스크에서 성능이 좋다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2014.png)

- 연산량에 따라 error가 얼마나 떨어지는지

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2015.png)

- 앞선 결과들은 제로샷 트랜스퍼가 가능한지 제너럴라이션이 얼마나 되었는지를 확인하고자 하는 부분이었다면 이번에는 랭귀지를 같이 학습함으로써 비주얼 리프리젠테이션이 얼마나 좋은 성능을 내는지를 평가를 하는 부분
- left :  12dataset
- right : 27 dataset
- → vit 기반 clip 성능이 가장 좋음 / 27 개는 clip 성능 압도적 → 다른 모델들 보다 generality 하다
- GFLOPS/image : cost와 관련, 즉 모델의 사이즈와 관련
- 아래 있는 모델들은(기존 모델들) 모델 사이즈가 커짐에 따라 성능이 떨어진다/saturation 되는 것을 시사
- 그리고 vit clip 봤을때, large model로 갈 수록 patch size가 감소 → vit에서 patch 사이즈를 줄이면 계산 연산량 증가함을 볼 수 있다
- 그리고 best 모델은 patch 14에 input 336*336 → 해상도가 더 좋은 이미지 사용 → 이렇게 보면 해상도 키워서 더 촘촘히 자르면 성능이 올라간다
- 생각해보면 당연해보임 → 그럼 ocr은 어떨까? → patch 작아지면 글자를 너무 크게 보지않을까?

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2016.png)

- efficientnet L2 noisy student
- 앞에서 제로샷 성능을 비교했을때는 뭔가 제너럴한 이미지에 대해서는 상당히 좋고 파인그레인즈 좀 더 세부적인 특화된 그런 이미지들에 대해서는 이제 성능이 좋지 않은 것을 발견 → 이 표에서는 비주얼 리프레젠테이션에 대해서 로지스틱 리그레이션을 수행한 이 결과는 대체적으로 그 클립의 성능이 더 좋은 것이 볼 수 있습니다 그래서 기존에 쏘타모델보다 비주얼 리프레젠테이션이 좀 더 제너럴하게 다양하게 이제 학습이 되었다라고 생각
- 여기서 cifar10, cifar100 는 low resolution이라 성능이 떨어지는것 같다라고 추측

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2017.png)

- 데이터셋에 변화가 주어졌을 때 인식하는 성능이 얼마나 로버스트한지 평가를 한 표
- imagenetV2: 더 복잡한 이미지
- imagenet R : 캐릭터
- objectnet : 다양한 환경에 존재하는 물체
- sketch : 물체에 대한 스케치 이미지
- imagenet A : 다양한 구도에서 찍힌 물체의 이미지

- x axis : 일반 data / y axis : 변화된 data → 왼쪽에 있다 일반화 성능이 높다
- Standard ImageNet training 보다 Zero-Shot CLIP가 높은 위치에 존재한다 → 더욱 roburst하다 → zero shor 이  더 genralize하다
- 

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2018.png)

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2019.png)

- zero shot 성능이 많이 좋다

→ 파인튜닝을 잘못하면 성능을해진다

→ 일반화시킬 수 있는 성능을 잘못하면 해칠 수 있다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2020.png)

→ ai가 못찾는 클래스는 사람도 못찾는다

→ zero shot  → one shot 찾는 방식이 다르다

![Untitled](contrastive%20language%20image%20pre-train/Untitled%2021.png)

→ 컴퓨터 비전 분야에서 task agnostic 한 objective를 훈련시킬 수 있을까에 대한 모티베이션을 가지고 시작된 논문

- task agnostic objective : 모델하나를 훈련시켜놓고 그걸 여기저기 다 갖다 쓸 수 있을까라는 얘기
- 그래서 zero shot transfer로 이어지고 그런 모델을 훈련시키려고 생각을 해보니 뭔가 좀 더 슈퍼비전을 받아야 될 거 같고 그러면 그 슈퍼비전을 어디서 받을까라고 생각해보니  이미지의 데이터에 보니까 다 데이터가 하나씩 붙어 있는 거예요  그래서 그런 애들로부터 같이 슈퍼비전을 받아 가지고 학습을 한다면 조금 더 비전 태스크를 잘 task agnostic objective 하게 할 수 있지 않을까라는 가설을 가지고 contrastive learning 을 통해 가지고 훈련 학습을 시켰던 것
- 텍스트 디스크립션과 이미지를 같이 컨트롤 하시는 러닝을 시켰더니 굉장히 제너럴리티가 개선이 되는 효과를 보았다
- 이거를 다양한 관점에서 다양한 각도에서 분석해 보았다.

⇒ 결론 

라지 스케일 랭귀지 비전 데이터셋을 가지고 굉장히 일반화된 비전 인코더 모델을 만드는 일을 한 겁니다 그래서 그 과정에서 텍스트 인코더의 도움을 받았던 거고요 그래서 앞으로는 resnet/vit 을 어떤 태스크에 맞게끔 파인 튜닝을 시켜서 사용하기보다는이 클릭 같은 모델을 크게 하나 구워 놓고 그걸 가지고 여기저기 사용하는 방식이 좀 더 보편화되지 않을까

[](https://arxiv.org/pdf/2207.07635.pdf)

→ 클립 논문에서 조금 아쉬운 부분이 어떤 부분이냐면이 클립 훈련을 어떻게 시켰느냐에 대한 얘기가 사실 잘 안 나와 있어요 그러니까 우리는 모델 아키텍처를 이렇게 두고 이런 wit라는 데이터셋을 뭐 어떤 식으로 쿼리를 통해 가지고 구해 가지고 대용량 학습을 시켰더니 잘 되더라 이런 얘기밖에 안 나와 있는데 그 디테일에 대해서 나온 논문

→클립을 학습시킬 때는 데이터 스케일이 굉장히 중요 

만약에 데이터 스케일이 받쳐주지 않으면 아까 얘기했던 이미자체에 달려있는 랭귀지 디스크립션 자체는 오히려 모델의 성능을 해칠 수 있다는 얘기