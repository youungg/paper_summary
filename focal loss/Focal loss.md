# Focal loss

**목차**

1. Focal Loss의 필요성
2. Focal Loss
    1. Cross Entropy Loss를 안쓰고 Focal Loss를 쓰는 이유
    2. Balanced Cross Entropy Loss를 안쓰고 Focal Loss를 쓰는 이유
    3. Focal Loss 적용
3. RetinaNet = Focal Loss + FPN
    1. FPN(Feature Pyramid Network)

---

![Untitled](Focal%20loss/Untitled.png)

                                                             *Focal loss 논문*

- Object Detection : 여러 object들을 Bounding Box를 통해 Localization(위치를 찾고) 하고, Classification(어떤 물체인지 분류) 하는 작업
- Object Detection의 2가지 종류의 알고리즘
    - R-CNN 계열의 two-stage detector
    - YOLO, SSD 계열의 one-stage detector
    - two-stage detector : localization을 한 다음에 classification이 순차적으로 이루어진다.
    - one-stage detector : localization과 classification을 동시에 처리한다.
    - 정확도 성능으로는 two-stage detector가 좋지만 연산 속도가 오래 걸리는 단점이 있다.

---

# **1. Focal Loss의 필요성**

![Untitled](Focal%20loss/Untitled%201.png)

                                    *대부분의 이미지는 Background Example들이 많다*

Focal Loss는 one-stage detector의 정확도 성능을 개선하기 위하여 고안되었다. one-stage detector가 two-stage detector에 비하여 가지고 있는 문제점은 학습 중 **Class imbalance(클래스 불균형) 문제가 심하다는 것**이다.

예를 들어, 학습 중 배경에 대하여 박스를 친 것과 실제 객체에 대하여 박스를 친 것의 비율을 살펴보면 압도적으로 배경에 대하여 박스를 친 것이 많다. 학습 중에서 배경에 대한 박스를 출력하면 오류라고 학습이 되지만 그 빈도수가 너무 많다는 것이 학습에 방해가 된다는 뜻이다. (SSD에서는 학습 시 한 이미지 당 만개 이상의 background에 대한 박스가 있다.)

이와 같은 문제점이 발생하는 이유는 **dense sampling of anchor boxes** (possible object locations)로 알려져 있다. 예를 들어 RetinaNet에서는 각각의 pyramid layer에서 anchor box가 수천개가 추출된다.

정리하면 이와 같은 클래스 불균형 문제는 다음 2가지 문제의 원인이 된다.

1. 대부분의 Location은 학습에 기여하지 않는 easy negative이므로 (detector에 의해 background로 쉽게 분류될 수 있음) **학습에 비효율적**이다.
2. easy negative 각각은 높은 확률로 객체가 아님을 잘 구분할 수 있다. 즉, 각각의 loss 값은 작다. 하지만 비율이 굉장히 크므로 전체 loss 및 gradient를 계산할 때, **easy negative의 영향이 압도적으로 커지는 문제가 발생**한다.

이러한 문제를 개선하기 위하여 **Focal Loss** 개념이 도입 되었다.

Focal Loss는 간단히 말하면 Cross Entropy의 클래스 불균형 문제를 다루기 위한 개선된 버전이라고 말할 수 있으며 **어렵거나 쉽게 오분류되는 케이스에 대하여 더 큰 가중치를 주는 방법**을 사용한다. (객체 일부분만 있거나, 실제 분류해야 되는 객체들이 이에 해당한다.) 반대로 **쉬운 케이스의 경우 낮은 가중치를 반영**한다. (background object가 이에 해당한다.)

---

# **2. Focal Loss**

---

# **2.1. Cross Entropy Loss를 안쓰고 Focal Loss를 쓰는 이유**

CE loss의 문제는 **모든 sample에 대한 예측 결과를 동등하게 가중치를 둔다**는 점이다. 이로 인해 어떠한 sample이 쉽게 분류될 수 있음에도 불구하고 작지 않은 loss를 유발하게 된다. 많은 수의 easy example의 loss가 더해지면 보기 드문 class를 압도해버려서 학습이 제대로 이뤄지지 않게 된다.

이진 분류에 대한 Cross Entropy Loss는 다음과 같은 식을 따른다.

![Untitled](Focal%20loss/Untitled%202.png)

                                                         *Cross Entropy Loss*

- y [1,−1] : ground truth class
- p [0,1] : 모델이 y=1 이라고 예측한 확률

예를 들어 다음과 같은 2가지 경우를 살펴보자.

첫번째는 Foreground 케이스이며, y = 1 이라고 하고 p = 0.95 라고 가정한다.

두번째는 Background 케이스이며, y = 0 이라고 하고 p = 0.05 라고 가정한다.

![Untitled](Focal%20loss/Untitled%203.png)

                                               *Cross Entropy Loss if p=0.95*

p = 0.95 일때의 Foreground 케이스를 살펴보면 Foreground인 객체에 대하여 높은 확률인 0.95로 잘 분류하였고, 그 결과 Loss가 0.05로 작은 것을 알 수 있다.

이와 유사하게 p = 0.05의 Background 케이스를 살펴보면 Background임에 따라 낮은 확률인 0.05로 잘 분류하였고 그 결과 Loss가 0.05로 작은 것을 알 수 있다.

문제가 없어보이지만 여기서 발생하는 문제점은 **Foregound 케이스와 Background 케이스 모두 같은 Loss 값을 가진다는 것**에 있다.

왜냐하면 Background 케이스의 수가 훨씬 더 많기 때문에 같은 비율로 Loss 값이 업데이트되면 Background에 대하여 학습이 훨씬 많이 될 것이고, 이 작업이 계속 누적되면 **Foreground에 대한 학습량이 현저히 줄어들기 때문**이다.

---

# **2.2. Balanced Cross Entropy Loss를 안쓰고 Focal Loss를 쓰는 이유**

Cross Entropy 케이스에서 발생하는 문제인 Foreground와 Background 케이스의 비율이 다른 점을 개선하기 위하여 Cross Entropy Loss 자체에 비율을 보상하기 위한 **weight(가중치 파라미터)를 추가로 곱해주는 방법**을 사용할 수도 있다.

예를 들어 Foreground 객체의 클래스 수와 Background 객체의 클래스 수 각각의 역수의 갯수를 각 Loss에 곱한다면, 클래스 수가 많은 Background의 경우 Loss가 작게 반영될 것이고 클래스 수가 적은 Foreground의 경우 Loss가 크게 반영될 것이다.

이와 같이 **각 클래스의 Loss 비율을 조절하는 weight**(wt) 를 곱해주어 imbalance class 문제에 대한 개선을 하고자 하는 방법을 Balanced Cross Entropy Loss 라고 한다.

일반적으로 0<wt<1 범위의 값을 사용하며 식으로 표현하면 다음과 같다.

![Untitled](Focal%20loss/Untitled%204.png)

                                                   *Balanced Cross Entropy Loss*

Cross Entropy Loss의 근본적인 문제가 Foreground 대비 Background의 객체가 굉장히 많이 나오는 class imbalance 문제에 해당하였다. 따라서 Balanced Cross Entropy Loss의 weight를 이용하면 weight에 대한 값의 조절을 통해 해결할 수 있을 것으로 보인다. 즉, **Foreground의 weight는 크게, Background의 weight는 작게 적용하는 방향으로 개선**하고자 하는 것이다.

하지만 이와 같은 방법에는 문제점이 있다. 바로, **Easy/Hard example 구분을 할 수 없다는 점**이다.

**단순히 갯수가 많다고 Easy라고 판단하거나 Hard라고 판단하는 것에는 오차가 발생할 수 있다.**

> Hard example은 모델이 예측하기 어려운 sample을 의미한다. 예를 들어, 실제로는 negative인데 positive라고 잘못 예측하기 쉬운 데이터입니다. 반면에 Easy example은 모델이 예측하기 쉬운 sample을 의미한다.
> 

다음과 같은 예를 살펴보자.

0.95의 확률로 Foreground 객체라고 분류한 Foreground 케이스에 weight 0.75를 주는 경우와,

0.05의 확률로 Background 객체라고 분류한 Background 케이스에 weight 0.25를 주는 경우를 살펴보자.

![Untitled](Focal%20loss/Untitled%205.png)

앞에서 설명한 바와 같이 통상적으로 Background 객체의 수가 많으므로 더 낮은 Loss를 반영하기 위해 더 작은 weight를 반영하도록 하였다. 그리고 식을 살펴보면 별 차이가 안난다. Easy/Hard Example에 대한 반영은 거의 없다.

---

# **2.3. Focal Loss 적용**

![Untitled](Focal%20loss/Untitled%206.png)

                                                             *Focal Loss*

Focal Loss의 핵심 아이디어는 다음과 같다. 모델 입장에서 쉽다고 판단하는 example에 대해서 모델의 출력 확률(confidence)인 **pt**가 높게 나올테니 **(1-pt)ᵞ**를 CE에 추가해줌으로써 **높은 확신에 대해 패널티**를 주는 방법이다. Focal loss는 **modulating factor (1−pt)ᵞ** 와  ᵞ 를 CE에 추가한 형태를 가진다. 여기서 ᵞ 를 **Focusing parameter** 라고 하며 **Easy Example에 대한 Loss의 비중을 낮추는 역할**을 한다.

반대로 어려워하고 있는 example에 대해선 pt가 낮게 나올테니 (1-pt)ᵞ 가 상대적으로 높게 나올 것이다!

ᵞ 가 높을 수록 ,(1-pt)가 작을 수록, Loss는 더 작아진다. (확신이 높은 example은 패널티를 더 받음)

이처럼 Focal Loss는 **Easy Example의 weight를 줄이고 Hard Negative Example에 대한 학습에 초점을 맞추는 Cross Entropy Loss 함수의 확장판**이라고 말할 수 있다.

Cross Entropy Loss 대비 Loss에 곱해지는 항인 (1−pt)ᵞ 에서 ᵞ > 0 의 값을 잘 조절해야 좋은 성능을 얻을 수 있다.

추가적으로 전체적인 Loss 값을 조절하는 ɑ값 또한 논문에서 사용되어 ɑ, ᵞ 값을 조절하여 어떤 값이 좋은 성능을 가졌는지 보여준다. 식은 아래와 같고 논문에서는 ɑ = 0.25, ᵞ = 2 를 최종적으로 사용했다.

![Untitled](Focal%20loss/Untitled%207.png)

위 그래프는 ᵞ 가 0~5 까지 변화할 때의 변화를 나타낸다. ᵞ = 0 일 때, Cross Entropy Loss와 같다. ᵞ 가 커질수록 Easy Example에 대한 Loss가 크게 줄어들며 Easy Examle에 대한 범위도 더 커진다.

서로 다른 ᵞ 값에 따른 loss는 위의 도표를 통해 확인할 수 있다. 위의 도표에서 파란색 선은 CE를 의미한다. 파란색 선은 경사가 완만하여 pt가 높은 example과 낮은 example 사이의 차이가 크지 않다는 것을 확인할 수 있다. 반면 Focal loss는 focusing parameter ᵞ 에 따라서 pt가 높은 example과 낮은 example 사이의 차이가 상대적으로 크다는 것을 확인할 수 있다.

**즉, y = 1(객체)인 class임에도 pt가 낮은 경우와, y = -1(배경)임에도 pt가 높은 경우에 Focal loss가 높다.**

반대의 경우에는 down-weight되어 loss값이 낮게 나타난다.

이를 통해 Focal loss의 두 가지 특성을 확인할 수 있다.

**1) pt와 modulating factor (1−pt)ᵞ 와의 관계**

example이 잘못 분류되고, pt가 작으면, modulating factor는 1과 가까워지며, loss는 영향을 받지 않는다.

반대로 pt 값이 크면 modulating factor는 0에 가까워지고, 잘 분류된 example의 loss는 down-weight된다.

**2) focusing parameter ᵞ 의 역할**

focusing parameter ᵞ 는 easy example을 down-weight하는 정도를 부드럽게 조정한다. ᵞ = 0인 경우, focal loss는 CE와 같으며, ᵞ 가 상승할수록 modulating factor (1−pt)ᵞ 의 영향력이 커지게 된다. 논문에서는 실험 시 ᵞ = 2일 때 가장 좋은 결과를 보였다고 한다.

![Untitled](Focal%20loss/Untitled%208.png)

위 그림을 살펴보자.

빨간색 케이스의 경우 Foreground Example의 문제이며, 초록색의 경우 Background Example(Object일 확률 값이 0.8 이상부터는 모델이 잘 맞추는)의 문제이다. (여기서 ɑ = 1, ᵞ = 1 이다.)

![Untitled](Focal%20loss/Untitled%209.png)

이제 Background Example일 때, CE에서 FL로 Loss값의 변화를 보면 0.1 에서 0.01로 내려갔다. 반면에 Foreground Example 일때는 2.3 에서 2.1로 내려갔다. Loss 값 숫자 자체만 보면 Foreground Example의 Loss 차이값이 더 크지만 Loss값이 의미하는 바를 생각해본다면 오히려 Background Example**이 훨씬 더 많이 감소한 것이 된다.**

Background Example은 기존에 90%의 확률로 예측했다고 가정하면 Focal Loss를 사용함으로써 99%의 확률로 예측한 것이 되므로 결국 잘 예측하는 데이터를 훨씬 잘 예측하도록 Loss 값을 매우 낮게 나오도록 계산하면서 해당 데이터는 학습을 더 하지 않아도 잘 맞출 것이라고 모델이 인지하게 된다.

반면에 Foreground Example은 2.3에서 2.1로밖에 낮아지지 않았으므로 해당 데이터에 대한 Loss 값이 여전히 높음을 의미하므로 Foreground Example에 더 집중해 학습을 하는 것이 좋겠다라고 모델이 인지하게 된다.

Hard 케이스 보다 **Easy 케이스의 경우 weight가 상대적으로(절대적X) 더 많이 떨어짐을 통하여, 기존에 문제가 되었던 수많은 Easy Negative 케이스에 의한 Loss가 누적되는 문제를 개선**한다.

이러한 원리로 인해 **Focal Loss를 사용**하면 Cross-Entropy를 사용할 때보다 **모델이 학습할 때 오분류하는 데이터에 더 집중해서 학습**하도록 할 수 있게 해준다.

```python
설명하다
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_lossPYTHON
```

---

# **3. RetinaNet = Focal Loss + FPN**

---

# **3.1. FPN(Feature Pyramid Network)**

![Untitled](Focal%20loss/Untitled%2010.png)

                                                              *RetinaNet Architecture*

**FPN이란, 서로 다른 크기의 객체들을 효과적으로 탐지하기 위해 Bottom-Up & Top-Down 방식으로 추출된 Feature Map들을 Lateral Connection 하는 방식**이다.

주목할 부분은 Top-Down인 FPN을 진행할 때 **Bottom-Up 할 때의 정보를 넘겨주기 위해 1 by 1 Convolution으로 Skip Connection을 수행해준다는 점**이다. 이것이 방금 언급했던 Lateral Connection이 된다. 또한 **Top-Down 시 Feature Map 사이즈를 유지시켜주기 위해 Convolution Layer 마다 사이즈를 2배 Up-sampling** 해준다.

그리고 **Top-Down 할 때의 각 Feature Map 마다 Object Detection을 수행**해 예측한다. 이렇게 하면 여러가지의 예측된 Bounding Box가 생겨날 것이다. 마지막으로 Ground Truth와 IoU를 비교하면서 최적의 Bounding Box만을 남기기 위해 NMS(Non-Max Suppression)을 수행해주어 최종적인 Object Detection을 수행해준다.

참고로 Top-Down인 FPN을 수행해줄 때 각 Feature Map 마다 Predict를 해주는데 여기에 3 by 3 Convolution을 추가해준다. 이렇게 하는 이유는 **Aliasing 현상을 막기 위함**이라고 한다. Aliasing 현상이란, **서로 다른 Feature Map들이 섞이면 자신들만의 특성이 사라지는데, 이 때 3 by 3 Convolution을 취하면 이 Aliasing 현상을 어느 정도 완화**시킬 수 있다고 한다.

![Untitled](Focal%20loss/Untitled%2011.png)

                                                        *RetinaNet Process*

---

Ref.

Focal Loss for Dense Object Detection 논문 : [https://arxiv.org/pdf/1708.02002.pdf](https://arxiv.org/pdf/1708.02002.pdf)

Focal Loss : [https://gaussian37.github.io/dl-concept-focal_loss/](https://gaussian37.github.io/dl-concept-focal_loss/)

RetinaNet : [https://herbwood.tistory.com/19](https://herbwood.tistory.com/19)

FPN : [앎의공간](https://techblog-history-younghunjo1.tistory.com/191?category=1031745)

Computer Vision 용어 정리 : [https://ganghee-lee.tistory.com/33](https://ganghee-lee.tistory.com/33)