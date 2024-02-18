# segmentation anything 발표자료

# Motivation

- 최근 LLM이 높은 zero shot/ few shot generalization  성능을 보인다
- LLM과 같이 대량의 데이터셋을 pretrain 하고 downstream  task에 대해 높은 zero shot generalization 성능을 보이는 모델을 foundation 모델이라고 부름
- 컴퓨터 비전 분야에서고 clip, align 과 같이 vision laguage dataset으로 foundation model을 만들려는 시도가 있었다

- 하지만 컴퓨터 비전 분야에서는 vision language task를 제외하고도 아직 풀어야하는 다양한 task가 존재
    - image segmentation
    - 3D reconstruction
    - high resolution
- 위 task들은 foundation model을 학습하기위한 대량의 이미지 데이터셋을 구축하기 어렵다

# objective

- image segmentation을 위한 foundation model을 만들어보자

# foundation model for inage segmentation

- 목표달성을 위한 세가지 질문
    1. what task will enable zero shot generalization
    2. whay os yje corresponding model architecture
    3. whay data can power this task and model

# segment anything task

- NLP에서는 foundation model을 학습하기위해 next token prediction task 정의

## promptable segmentation taask

- 이미지와 어떤 prompt가 주어졌을때 유효한 mask를 반환

![Untitled](segmentation%20anything/Untitled.png)

## prompt

- 이미지에서 분할할 대상을 지정하는것
- points, bbox, mask, 심지어 text가 될 수도있다

![Untitled](segmentation%20anything/Untitled%201.png)

## 유효한 mask

- ambigious(모호한) prompt가 주어졌을때도 합리적인 mask를 출력

![Untitled](segmentation%20anything/Untitled%202.png)

# segment anything data engine &dataset

- next token prediction task의 경우 웹상에서 텍스트 데이터를 대규모로 얻을수 있다
- promprable segmentation task 학습을 위한 segmentation mask는 구하기 어렵다

→ 직접 만듬

## 대규모 mask data 를 획득하기위한 3가지 stage

1. assisted manual stage
2. semi automatic stage
3. fully automatic stage

## 1. assisted manual stage

- 공개된 segmentation dataset을 이용해 sam을 초기 학습
- 전문 annotator들이 웹기반의 인터페이스에서 초기 학습된 sam을 이용해 데이터 생성
- 새로 취득한 data로만 점진적 모델학습 진행
- 120K 이미지로 부터 4.2M mask 취득

## 2. semi automatic stage

- mask의 종류를 다양화 하는 것을 목표로 함
- 1단계에서 학습된 sam을 이용해 신뢰도 높은mask를 작업화면에 표시
- annotator들은 그외 object를 작업
- 새로 취득한 data로 점진적 모델 학습
- 180K 이미지로부터 5.9M mask를 취득

## 3. fully automatic stage

- 완전 자동화된 annotation 단계
- 2단계까지 학습된 모델에 32 * 32 regular grid point 를 입력하여 mask 획득
- IOU 값이 높은 mask 만 남김
- 중복된 mask를 제거등 후처리 작업 수행
- 11M 이미지로부터 1.1B mask를 취득

## SA-1B 데이터셋

- annotator가 만들지 않고 sam이 자동으로 만든 데이터만 포함

![Untitled](segmentation%20anything/Untitled%203.png)

# segment anything model

## 목표

1. promptable segmentation task
2. real world 에서 interative 하게 사용가능 (속도)

## 목표를 위한 모델의 제약 조건

1. 유연한 prompt 지원
2. prompt의 모호함에 대한 대처
3. mask를 real time으로 연산

![Untitled](segmentation%20anything/Untitled%204.png)

- 강력한(layer 가 깊은) image encoder가 image embedding을 계산
- prompt encoder가 prompt의 embedding을 계산
- 두 embedding 정보를 loghtweight mask decoder에서 결합하여 mask 예측

![Untitled](segmentation%20anything/Untitled%205.png)

- 고해상도 이미지를 처리하기 위해
- masked autoencoder(MAE)로 pre training 한 vision transformer(vit) 구조를 사용
- 동일한 이미지에 대한 embedding은 다른 prompt에 재사용할 수 있다
    - 하나의 이미지에 대해서 한번만 embedding을 계산

![Untitled](segmentation%20anything/Untitled%206.png)

- sparse prompt : points, boxes, text
    - 이미지에서 segment 할 대상을 지정할 정보
- Dense prompt : masks
    - 이미지와 공간적으로 대응되는 정보
    - image embedding과 element wise로 더해진다
    
    → 이전 프롬프트를 활용하기 위해 사용
    
    → multi point 같은 경우, point가 일그러지는 현상이 나타날 수 있어, 이전 mask를 재사용
    

![Untitled](segmentation%20anything/Untitled%207.png)

- image embedding 과 prompt embedding을 받아 mask 를 출력
- prompt encoder + mask decoder는 50ms 이내에 mask를 예측
- 모호성에 대응하기 위해 여러개의 mask를 예측하도록 설계 (whol, part, sub part)

![Untitled](segmentation%20anything/Untitled%208.png)

# applications : one click segmentation

![Untitled](segmentation%20anything/Untitled%209.png)

# applications : Everything

![Untitled](segmentation%20anything/Untitled%2010.png)

![Untitled](segmentation%20anything/Untitled%2011.png)

1. 1024개의 지점을 64개씩 16번 배치 수행
2. mask와 iou_prediction 추론
3. mask를 필터링
    1. threshold 이하의 mask를 제거
    2. 겹쳐지는 mask 들을 제거
4. 각 mask의 hole과 island를 제거

> [https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)
> 

# application : sam with clip

1. everything 으로 모든 mask 찾기
2. clip으로 query text와 각 mask간 유사도 측정
3. 유사도 점수가 높은 mask만 남김

![Untitled](segmentation%20anything/Untitled%2012.png)

![Untitled](segmentation%20anything/Untitled%2013.png)

# application : grounded segmentation anything

![Untitled](segmentation%20anything/Untitled%2014.png)

1. grounding dino 로 open vocabulary object detection 수행
2. sam으로 bounding box내의 object에 대한 segmentation 수행

→text 입력시 text에 해당하는 모든것을 object detection

> https://github.com/IDEA-Research/Grounded-Segment-Anything
> 

# application : inpaint anything(with stable diffusion)

1. sam으로 원하는 object를 segmentation
2. stable diffusion 등 으로 이미지 재생성

> https://github.com/geekyutao/Inpaint-Anything
> 

# application : sam track

1. sam 이 첫번째 프레임의 segmentation 수행
2. DeAOT로 다음 frame의 mask를 예측

|https://github.com/z-x-yang/Segment-and-Track-Anything

→ 첫 프레임 segmentation 후, 같은 object는 tracking, 새로운 object가 나오면 걔네는 다시 segmentation

# application : awesome segment anything

https://github.com/z-x-yang/Segment-and-Track-Anything

https://github.com/JerryX1110/awesome-segment-anything-extensions

실습

[](https://github.com/MrSyee/SAM-remove-background/blob/main/jupyternotebook/sam_click.ipynb)