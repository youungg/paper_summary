# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

## **1. 들어가며**

이번 글에서는 인공지능 분야에서 큰 주목을 받은 CLIP 논문에 대해 자세히 살펴보려고 합니다. 이 논문은 기존의 컴퓨터 비전 모델과 언어 모델의 한계를 넘어서려는 새로운 시도를 담고 있어, 많은 관심을 받았습니다. 2023년 현재에는 GPT-4, LLaVA등 다양한 Large Multimodal Model (LMM)들이 발표되고 있는데요. CLIP은 이러한 LMM들의 시초 연구라고도 할 수 있는 모델입니다. 이 글을 통해 CLIP 논문의 핵심 내용을 쉽고 자세하게 설명하고자 합니다. 여러분이 이 글을 통해 CLIP의 중요성과 그 의미를 깊이 이해하시길 바랍니다.

우선, 기존의 컴퓨터 비전 모델들은 주로 이미지만을 학습하여 성능을 향상시켜왔습니다. 하지만 이러한 접근 방식은 모델의 강건함과 일반화 능력에 한계를 가지고 있었습니다. 반면, 언어 모델들은 대규모 언어 데이터를 학습하며 급속도로 발전해왔습니다. 이러한 배경에서 CLIP 논문의 저자들은 언어 모델처럼 대규모 데이터셋을 학습하는 방식이 이미지 인식 분야에서도 중요한 역할을 할 수 있다고 생각했습니다. 이러한 생각은 기존의 접근 방식에 대한 새로운 시각을 제공합니다.

이 글에서는 CLIP 논문의 주요 내용을 여러분이 쉽게 이해할 수 있도록 다음과 같은 순서로 설명해드릴 예정입니다. 먼저, 기존 방법의 문제점을 살펴보면서 기존의 비전 모델과 언어 모델이 어떻게 발전해왔으며, 그 과정에서 나타난 한계점들을 이해할 것입니다. 이 부분은 CLIP 논문의 배경과 필요성을 이해하는 데 중요한 기초가 됩니다.

그 다음으로는 CLIP이 어떻게 이미지와 자연어를 결합하여 새로운 학습 방법을 제안했는지, 그리고 이를 통해 어떤 문제를 해결하려고 했는지 설명하겠습니다. 이 부분에서는 CLIP이 어떻게 대규모 이미지-자연어 쌍 데이터셋을 제작하고, contrastive learning 방법을 활용하여 학습했는지에 대해 자세히 살펴볼 것입니다. 또한, zero shot prediction 방법에 대해서도 쉽게 설명하겠습니다.

실험 결과 부분에서는 CLIP 모델이 실제로 어떤 실험을 거쳤고, 그 결과가 어떠했는지 살펴볼 것입니다. 여기서는 zero shot transfer, representation learning, 그리고 다양한 데이터셋에 대한 강건함을 보이는 실험 결과들을 분석해보겠습니다. 이 부분은 CLIP 모델의 실제 성능과 그 한계를 이해하는 데 중요한 역할을 합니다.

이어서 토론 부분에서는 사람과 CLIP의 차이점, 특히 인지 능력과 학습 방식에서의 차이점을 탐구해보겠습니다. 이 부분은 CLIP 모델이 인간의 인지 능력과 어떻게 다른지, 그리고 그 차이가 모델의 성능에 어떤 영향을 미치는지 이해하는 데 도움이 될 것입니다. 또한, CLIP 모델의 장점과 단점을 구체적으로 분석해보고, 마지막으로 CLIP이 인공지능 분야에 어떤 의미를 가지는지, 그리고 앞으로의 발전 가능성에 대해 이야기하겠습니다.

이 글을 통해 여러분이 CLIP 논문의 핵심 내용을 이해하고, 이 분야의 발전에 대한 통찰을 얻을 수 있기를 바랍니다. CLIP 논문은 단순히 새로운 기술을 제시하는 것을 넘어, 인공지능 분야에서의 새로운 방향성을 제시하는 중요한 작업입니다. 이 글을 통해 그 중요성을 함께 공유하고자 합니다. 중요한 참고 자료는 본문에 표시하고 마지막 챕터인 참고 자료편에 링크를 첨부했으니, 해당 링크를 참고해주세요.

## **2. 기존 방법의 문제점**

먼저 CLIP 이전에 존재했던 방법들의 문제점을 생각해보겠습니다. CLIP이 발표된 시점이 2021년이니까, 이 시점을 기점으로 크게 두가지 카테고리로 구분하여 기존 방법들의 발전 방향을 생각해보겠습니다. 살펴볼 두 가지 카테고리는 Vision Model과 Language Model입니다. 왜냐하면 이 주제들이 CLIP이 가장 큰 영향을 받은 카테고리들이기 때문이죠.

먼저 2021년 당시 Vision Model의 발전 방향을 생각해볼게요. Vision Model은 전통적으로 이미지를 입력받아 어떻게 모델을 구성하면 더 좋은 표현을 학습하는지를 고민해왔습니다. Inception[1], ResNet[2] 등은 효율적인 모듈을 구성하여 깊은 모델을 만드는 방법을 고민했죠. SENet[3], BAM[4], CBAM[5] 등은 Attention 모듈을 적용하는 방법을 제안했습니다. 또 2021년 당시 Vision Model의 트렌드는 Transformer[8] 구조를 적용하는 것이었죠. 이러한 트렌드에 맞추어 발표된 모델이 ImageGPT[6], Vision Transformer[8] 였습니다. 이렇게 Vision Model은 날이 갈수록 발전해왔는데요. 하지만 이미지만 학습한 모델은 고질적으로 일반화 능력이 부족하고 작은 노이즈에도 취약한 약점을 보였습니다.

한편 Language Model은 Vision Model 보다 한 발 앞서 나아가는 형태로 발전해왔습니다. Vision Model이 Inception[1], ResNet[2] 등 다양한 CNN 모델을 발표하며 발전하던 시기, Language Model은 seq2seq 방식의 한계로 인해 크게 나아가지 못하고 있었죠. 이후 2017년 Transformer[8]의 발표를 기점으로 큰 변화가 생기는데요. 마치 Vision Model이 CNN 발명에 힘입어 크게 도약한것 처럼, Language Model도 Transformer[8] 구조의 발명에 힘입어 한 단계 도약하게 됩니다. Transformer[8]는 seq2seq 구조와 달리 긴 문장도 효과적으로 처리할 수 있기 때문이죠. 이후 GPT 시리즈들 (GPT-1[9], GPT-2[10], GPT-3[11]), BERT[12] 등 다양한 초거대언어모델 (LLM)이 발표되게 됩니다. 이때부터 Language Model은 LLM의 시대가 열리게 되죠.

자, 이 시점에서 CLIP 저자들은 이런 생각을 합니다. Language Model은 LLM의 시대를 맞아 급격하게 발전하고 있다고 했는데요. 그럼 Vision Model도 LLM과 같은 방향으로 간다면 한 단계 더 발전할 수 있지 않을까요? LLM에는 크게 두가지 조건이 있는데요. 첫 번째는 큰 모델이고, 두 번째는 큰 데이터입니다. 큰 모델은 차치하고라도 우선 데이터셋이라도 아주 크게 만들어서 학습한다면 지금의 Vision Model의 한계를 넘어설 수 있지 않을까요? 당시 Vision Model의 대표 데이터셋은 ImageNet 이었는데요. 데이터의 양이 적은편은 아니지만, 각 이미지에 대해 사람이 직접 Label을 달아놓은 형태의 데이터셋이죠. 그 말은 아무래도 데이터의 개수에 한계가 있다는 뜻이죠. 사람이 직접 Label을 달아서는 수집할 수 있는 개수에 한계가 있으니까요.

이렇게 CLIP 저자들은 기존 ImageNet으로 대표되는 Labeled Image Dataset으로 학습하는 Vision Model의 한계에 대해 고민합니다. 이제 다음 챕터에서부터는 CLIP 저자들이 어떻게 이 문제를 해결했는지 살펴보겠습니다.

## **3. CLIP**

이번 챕터에서는 CLIP 논문의 제안 방법을 살펴보겠습니다. 앞서 설명한 것 처럼 CLIP 저자들은 우선 데이터셋의 한계에 주목하는데요. 따라서 이러한 기존 이미지 데이터셋 구성의 한계를 어떻게 극복했는지 자세히 살펴보겠습니다. 또한 이렇게 대용량 Image-Text 데이터는 기존 Vision Model로 학습할 수가 없는데요. 어떻게 모델을 구성하여 효과적으로 Image-Text 데이터셋을 학습할 수 있었는지 살펴보겠습니다. 마지막으로 CLIP의 전매특허라 할 수 있는 Zero Shot Prediction은 어떤 방식으로 이루어지는지 살펴보겠습니다.

### **3-1. 자연어 Supervision 학습하기**

앞서 ImageNet으로 대표되는 Image Dataset은 한계가 있다고 설명했는데요. 따라서 CLIP 저자들은 우선 대용량의 데이터셋을 확보하기 위해 인터넷에서 데이터를 모으는 방법을 선택합니다. 하지만 이렇게 되면 ImageNet 처럼 Label을 달아줄 수가 없는데요. 인터넷상에서 모은 데이터는 이미지별로 Label이 달려있지 않을 뿐더러, Label이 있는 데이터를 찾는다고 해도 그 정확도를 보장할 수 없기 때문이죠. 또 애초에 대용량 데이터셋을 구축해야 한다는 전제가 있기 때문에 사람이 직접 Label을 달아준다는 생각이 모순이기도 하고요.

이러한 상황에서 CLIP은 어떤 방법을 선택했을까요? 바로 자연어 (Natural Language)를 Supervision으로 사용하는 방법을 선택합니다. Supervision은 지도, 지시, 감독 등의 뜻을 갖는 단어인데요. 모델에게 이미지를 설명해주는 역할이라고 이해할 수 있습니다. 기존 ImageNet으로 학습하는 모델들에게는 Label 정보가 Supervision이었겠죠. 그런데 이번에는 Label은 사용할 수 없으니, 인터넷상에서 이미지마다 달려 있는 자연어 문장을 그대로 Supervision으로 사용하자는 아이디어입니다.

![https://ffighting.net/wp-content/uploads/2023/11/image.png](https://ffighting.net/wp-content/uploads/2023/11/image.png)

그림1. CLIP 데이터셋

위 그림은 CLIP 논문에 나오는 데이터셋 예시 그림인데요. 왼쪽 이미지의 가장 위에는 강아지 이미지가 있습니다. 오른쪽 뭉터기 가장 위에는 이 이미지를 설명하는 자연어 (Pepper the aussie pup) 이라는 문장이 있고요. 이렇게 이미지와 이 이미지를 설명하는 문장은 인터넷상에서 아주 많이 존재하죠. CLIP은 이렇게 인터넷으로부터 자그마치 4억장의 이미지-자연어 매칭 데이터셋을 구축합니다.

### **3-2. 효율적인 학습 방법 선택하기**

대용량 데이터셋을 만드는것 까지는 좋았는데요, 이를 어떻게 학습해야 할까요? 기존 ImageNet 데이터셋처럼 Cross Entropy Loss로 학습할 수는 없습니다. 왜냐하면 자연어는 Label과 달리 특정 개수로 구분되지 않기 때문이죠. 이미지 마다 매칭되어 있는 설명 문장은 전부 다를거잖아요. 따라서 Softmax로 구분하는 방식의 학습 방법은 가능하지 않습니다.

이러한 고민의 끝에서 CLIP 저자들이 선택한 방법은 Contrastive Learning 입니다. Contrastive는 ‘대조하는’ 이라는 뜻을 가진 단어인데요. 따라서 Contrastive Learning은 매칭되는 데이터 Feature들끼리는 가까워지도록, 나머지 Feature들 끼리는 멀어지도록 학습하는 방법입니다. 데이터를 대조해가며 나랑 매칭되는 데이터는 가까워지도록, 다른 데이터는 멀어지도록 모델을 학습하는 방법이죠. 이러한 Contrastive Learning 학습 방법은 Self Supervised Learning에서 그 진가를 발휘했는데요. Label 정보가 없어도 어떠한 기준으로 나와 매칭되는지만 설정해주면 학습을 할 수 있기 때문이죠. 이렇게 Contrastive Learning을 사용한 대표적인 Self Supervised Learning 모델은 SimCLR[13]가 있습니다. 입력 이미지에 Augmentation을 적용하여 동일한 이미지 버전끼리는 가까워지도록, 다른 이미지 버전과는 멀어지도록 학습했죠. 놀라운건 이렇게 Label 정보 없이 학습했음에도 Label 정보로 학습한 모델에 버금가는 표현력을 학습했음을 실험적으로 증명했다는 점입니다.

![https://ffighting.net/wp-content/uploads/2023/11/image-1.png](https://ffighting.net/wp-content/uploads/2023/11/image-1.png)

그림2. CLIP Contrastive Learning

위 그림은 CLIP의 Contrastive Learning 방법을 설명한 그림입니다. 앞서 설명한 Contrastive Learning 방법이 그대로 적용된 모습을 볼 수 있습니다. 데이터셋은 이미지와 이 이미지에 매칭되는 자연어로 구성되어 있고요. 이미지는 Image Encoder로, 자연어는 Text Encoder로 Feature를 추출해줍니다. 이렇게 추출한 Image Feature는 초록색 사각형 (IN)으로, Text Feature는 보라색 사각형 (TN)으로 표현해 주었습니다. 이때 N은 배치 개수를 의미합니다.

그럼 총 N개의 Image Feature가 있고, 마찬가지로 N개의 Text Feature가 추출되어 있는 상황이죠. 이들 각각을 매칭해보면 총 NxN개의 조합이 나오는데요. Contrastive Learning은 나와 매칭되는 조합은 가까워지도록, 그 외의 조합은 멀어지도록 학습하는 방법이라고 했죠. 이때 가까워진다는 의미는 여기서는 두 Feature의 Cosine Similarity가 커지는 방향을 말합니다. 두개의 Feature가 공간상에서 가까운 각도에 위치할수록 Cosine Similarity는 큰 값을 갖기 때문이죠. 반대로 나머지 쌍과는 멀어지도록 모델을 학습해줍니다. 여기서 모델은 Image Encoder와 Text Encoder를 의미하죠.

여기까지가 CLIP 학습 방법의 가장 핵심 내용입니다. 정리해볼게요. 총 4억장의 Image-Text Pair 데이터셋을 학습할건데요. N개의 배치에 대해 Image Encoder, Text Encoder가 추출한 Feature들을 각각 N개씩 추출했습니다. 이 조합은 NxN개가 되고요. 이때 Encoder들은 어떤 방향으로 가중치를 업데이트할 거냐면, 본인 Image, Text Feature들끼리는 가까워지도록, 나머지 Feature들끼리는 멀어지도록 학습해줍니다.

### **3-3. 적절한 Encoder 선택하기**

앞서 살펴본 바와 같이 결국 CLIP 방법론의 핵심은 Image Encoder와 Text Encoder를 Contrastive Learning 방법으로 학습한다는 것인데요. 이 두 Encoder는 어떻게 구성해주었을까요?

먼저 Image Encoder는 다양한 Vision Model들이 가능합니다. 대표적으로 ResNet[2], ViT[7]가 있죠. ResNet[2]은 조금 더 표현 추출 능력을 강화하기 위해 마지막 Global Average Pooling 부분을 수정해줍니다. 여기에 Attention 모듈을 추가한 Attention Pooling으로 적용해줍니다. ViT[7]는 기존 구성 거의 그대로 사용해줍니다. 이렇게 5가지 종류의 ResNet[2]과 3가지 종류의 ViT[7]를 사용하여 실험을 진행해줄겁니다.

한편 Text Encoder는 Transformer[8]를 사용합니다. 마지막 Token에서 추출된 Feature를 Linear Projection 해주어 Image Feature와의 차원을 맞춰줍니다.

### **3-4. Zero Shot 예측 방법**

지금까지 CLIP 모델의 학습 방법을 살펴봤는데요. CLIP 모델의 가장 큰 특징이라고 한다면 역시 이미지-자연어 매칭쌍을 학습한 것이라고 할 수 있습니다. 기존 Vision Model과 달리 이미지만을 학습하지 않고 이미지-자연어 쌍을 학습했으니 뭔가 다른 기능도 가능할것 같은데요. CLIP의 가장 재미있는점중 하나는 바로 Zero Shot Prediction이 가능하다는 것입니다.

Zero Shot Prediction이 뭐냐면요, 말 그대로 한번도 학습하지 않은 문제를 맞추는 방법입니다. 이러한 기능은 기존 ImageNet으로 학습한 모델에서는 기대하기 어려웠는데요. 왜냐하면 Supervised Learning 방식으로는 학습하지 않은 클래스를 예측하는게 태생적으로 불가능하기 때문이죠. 1000개 클래스에 대해서만 구분하도록 학습했는데 갑자기 1001번째 클래스를 예측하라고 한다면 1001번째 클래스의 예측값은 엉뚱한 값이 나올 수 밖에 없겠죠.

그런데 재밌는건 CLIP은 Zero Shot Prediction이 가능합니다. 우선 어떻게 하는지 그 방법을 먼저 살펴볼게요.

![https://ffighting.net/wp-content/uploads/2023/11/image-2.png](https://ffighting.net/wp-content/uploads/2023/11/image-2.png)

그림3. CLIP Zero Shot Prediction

위 그림은 학습이 완료된 CLIP의 Zero Shot Prediction 방법을 설명한 그림입니다. CLIP 논문상에서 캡쳐한 그림인데요. 아래 그림을 보고 조금 더 자세히 살펴볼게요.

[data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==](data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==)

그림4. Zero Shot Prediction 방법

위 그림은 Zero Shot Prediction을 구성하는 Feature 연산 부분만 도식화한 그림입니다. 먼저 Zero Shot Prediction을 하려는 이미지 Feature를 추출해야겠죠. 이 Feature는 학습 완료된 CLIP의 Image Encoder를 사용하여 추출해주면 됩니다. 위 그림에서는 초록색 사각형에 해당합니다. 이제 이 그림이 어떠한 클래스에 해당하는지를 연산하기 위해 Text Feature를 추출해야 하는데요. 우리가 구분하고자 하는 클래스 종류가 N개라고 가정할게요. 위 그림의 예시에서는 plane, car, dog … bird에 해당합니다. 그럼 이제 이 클래스들을 설명하는 문장을 만들어줄거에요. ‘A photo of a plane’ 처럼요. 모든 N개의 클래스를 이런식으로 해당 클래스를 표현하는 문장으로 만들어주는거죠. CLIP이 이렇게 그림을 설명하는 문장을 학습했으니까요. 자 이제 N개의 문장을 각각 학습 완료된 CLIP의 Text Encoder로 Feature를 추출해줍니다. 위 그림에서는 N개의 보라색 사각형들에 해당합니다. 이제 마지막 단계로 추출한 Image Feature와 Text Feature들간의 Cosine Similarity를 측정해줍니다. 그럼 해당 이미지를 가장 잘 설명하는 Text Feature와의 Cosine Similarity 값이 가장 크게 나올겁니다. 왜냐하면 CLIP이 바로 그렇게 학습했기 때문이죠.

이러한 방법을 통해서 CLIP은 고정되지 않은 개수의 클래스에 대해 예측이 가능합니다. 이는 기존의 Label을 사용하여 이미지의 클래스를 구분하는 방식이 아닌, 이미지와 자연어의 정렬 (Align)을 학습한 덕분입니다. 이제 아래 챕터에서는 이러한 CLIP이 Zero Shot Prediction을 비롯한 다양한 성능을 비교한 실험 결과를 살펴보겠습니다.

## **4. 실험 결과**

이번 챕터에서는 CLIP의 실험 결과를 살펴보겠습니다. 크게 세가지 측면에서 실험 결과를 살펴볼건데요.

첫 번째는 Zero Shot Transfer 실험입니다. Zero Shot Prediction 방식과 Linear Probing 방식의 성능 비교를 통해 CLIP이 얼마나 좋은 표현을 잘 학습했는지 살펴보겠습니다.

두 번째는 CLIP으로 학습한 모델을 Linear Probing 방식으로 추가 학습을 하여 기존 모델들 대비 얼마나 좋은 표현을 학습했는지 살펴보겠습니다.

세 번째는 Robustness 정도를 살펴보겠습니다. 기존 Vision Model들은 좋은 표현을 잘 학습함에도 일반화 능력이나 강건함 (Robustness) 능력이 떨어지는 경향이 있는데요. 실험을 통해 CLIP은 이러한 측면에서 개선점이 있었는지 살펴보겠습니다.

### **4-1. Zero Shot Transfer**

먼저 Zero Shot Transfer 실험 결과를 살펴보겠습니다. 이번 실험에서 살펴보고자 하는건 결국 ‘CLIP이 얼마나 좋은 표현을 학습했는지’입니다. 이를 비교하기 위한 방법을 다양할텐데요. 이번에는 다양한 방법 중 Zero Shot Prediction 성능을 기준으로 비교를 합니다.

![https://ffighting.net/wp-content/uploads/2023/11/image-4.png](https://ffighting.net/wp-content/uploads/2023/11/image-4.png)

그림5. Zero Shot Transfer 성능 비교

위 그림은 각 데이터셋에 대해 CLIP의 Zero Shot Prediction 성능과 Linear Probe 성능을 비교한 그래프입니다. Linear Probe란 학습이 완료된 Encoder를 가져와 Supervised Learning으로 Classifier만 재학습해주는 방법인데요. 만약 Encoder가 좋은 표현을 많이 학습했다면 단순히 Classifier만 재조정 해주어도 높은 성능이 나올 것이라는 전제가 깔려있는 방법이죠.

결과를 보면 위의 절반에 가까운 데이터셋에서는 CLIP의 Zero Shot 성능이 Linear Probe 성능보다 좋은 모습을 볼 수 있습니다. 조금 더 구체적으로 살펴보면 세부적인 표현 학습이 필요한 Fine Grained Classification 데이터셋에서는 성능이 안좋고, 반대로 일반적인 표현 학습만으로 풀 수 있는 데이터셋에서는 성능이 좋은 모습을 보이고 있습니다. 이러한 결과는 매우 고무적이라고 할 수 있는데요. 왜냐하면 모든 데이터셋에서 좋은 결과를 낸 것은 아니지만 Label 데이터를 전혀 사용하지 않고도 Label 정보를 사용하여 학습한 동일한 모델보다 더 좋은 성능을 보여주고 있기 때문입니다.

![https://ffighting.net/wp-content/uploads/2023/11/image-5.png](https://ffighting.net/wp-content/uploads/2023/11/image-5.png)

그림6. Linear Probing 성능 비교

다음 실험은 CLIP과 다른 모델들의 Linear Probing 성능을 비교한 그래프입니다. x축은 Linear Probing에 사용한 클래스당 데이터 개수를 의미합니다. 사전 학습이 완료된 상태에서 몇개의 대표 데이터만을 사용하여 Classifier를 학습했을때 누가 더 성능이 좋은지를 비교한것인데요. 흥미로운 점은 우선 CLIP 모델이 모든 면에서 다른 모델들보다 Linear Probing 실험에서 좋은 성능을 보인다는 점입니다. 기존의 SimCLR[13], BiT 등 좋은 표현을 학습한다고 알려진 다른 방법들보다 좋은 표현을 학습한다는 점이 검증된것이죠. 또 주목해야 할 부분은 CLIP의 Zero Shot 성능입니다. CLIP의 Zero Shot 성능은 클래스당 4개의 데이터를 학습한 CLIP 모델과 비슷한 수준이고, 다른 모델들은 더욱 많은 데이터셋을 학습해야 낼 수 있는 수준의 성능입니다. 이를 통해 CLIP의 Zero Shot 성능이 상대적으로 얼마나 좋은지를 알 수 있습니다.

[data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==](data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==)

그림7. Linear Probing과 Zero Shot 성능 비교

위 그림은 CLIP 모델만을 사용하여 Zero Shot 성능과 Linear Probing 성능을 비교한 그래프입니다. 아무래도 동일한 사전학습을 한 모델에서는 Linear Probing의 성능이 더 좋은 모습을 보이고 있습니다. 하지만 몇몇 데이터셋에서는 Zero Shot 성능과 Linear Probing 성능이 거의 유사한 모습을 보이고 있습니다. 이를 통해 마찬가지로 CLIP 모델의 Zero Shot Prediction이 얼마나 강력한지를 알 수 있습니다.

### **4-2. Representation Learning**

하지만 이렇게 Zero Shot Prediction 성능으로 해당 모델이 좋은 표현을 학습했는지를 검증하는 방법은 익숙하지 않은데요. 보통은 Fine Tuning 성능과 Linear Probing 성능을 통해 학습한 표현력을 비교하기 때문이죠. 이에 CLIP에서는 Linear Probing 성능 비교 실험을 진행합니다. Zero Shot Prediction은 CLIP만의 특화된 기능이잖아요. 따라서 Zero Shot Prediction을 잘한다는 이유만으로 CLIP이 더 좋은 표현을 학습했다고 인정할 수 없으니, 기존 룰대로 테스트하자는거죠.

[data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==](data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==)

그림8. 모델별 Linear Probing 성능 비교

위 그림은 다양한 데이터셋에 대해, 다양한 모델들의 Linear Probing 성능을 비교한 그래프입니다. CLIP 방식으로 학습한 ViT[7], ResNet[2]과 기존 다른 방법들의 Linear Probing 성능을 비교하고 있습니다. 동일한 조건상의 모델끼리 비교했을때 CLIP 방식으로 학습한 모델의 Linear Probing 성능이 가장 좋은 모습을 볼 수 있습니다.

Linear Probing 테스트는 Feature Extractor 부분은 고정해 놓은채로 Classifier 부분만 재학습하여 하는 테스트입니다. 따라서 Linear Probing 성능은 해당 모델의 Feature Extractor가 얼마나 범용적이고 효과적인 표현을 학습했는지를 대표하는 평가방식으로 사용되어 왔습니다. 이러한 Linear Probing 테스트에서 CLIP 모델이 기존 모델들보다 좋은 성능을 냈다는 의미는 그만큼 이미지-자연어 쌍을 Contrastive Learning 으로 학습하는 방법이 우수함을 입증한다고 할 수 있습니다.

### **4-3. Robustness to natural distribution shift**

지금까지는 CLIP 모델이 기존 Vision Model들보다 ‘성능’ 이 더 우수함을 실험을 통해 확인했습니다. 이때의 성능이란 얼마나 더 많은 표현을 학습했는지를 의미하는데요. 대표적으로 방금 살펴본 Linear Probing 테스트를 예로 들 수 있습니다. 하지만 Vision Model들의 한계는 성능에 있지 않은데요. 이들 모델의 공통적인 한계는 Robustness가 현저하게 떨어진다는 점입니다. 예를 들어 학습한 데이터셋에서 노이즈가 조금 섞여 들어간다거나, 텍스쳐가 변한다면 성능이 아주 크게 하락하곤 했죠.

따라서 진정 우수한 Vision Model이라면 우수한 성능 뿐만 아니라 일반화 성능과 강건함 (Robustness)까지 갖추어야 할텐데요. CLIP 저자들은 이러한 관점에서 Robustness에 대한 실험도 진행합니다.

![https://ffighting.net/wp-content/uploads/2023/11/image-8-1024x416.png](https://ffighting.net/wp-content/uploads/2023/11/image-8-1024x416.png)

그림9. Robustness 실험 결과

위 그림은 ImageNet에 약간의 변형을 준 데이터셋에 대한 성능을 비교한 자료입니다. 기존 ImageNet과 달리 스케치 형태의 데이터이거나 Adversarial Attack이 추가된 형태등을 확인할 수 있습니다. 그리고 이러한 데이터셋들에 대해 기존의 ResNet[2]과 CLIP 모델의 Zero Shot 성능을 비교하고 있습니다.

우선 놀라운건 모든 데이터셋에 대해 기존 ResNet[2]보다 CLIP의 Zero Shot 성능이 더 좋은 모습을 보인다는 점인데요. 특히 기존 ImageNet의 특성과 달라지는 데이터일 수록 더욱 큰 성능 차이를 보였습니다. 예를 들어 ImageNet으로 학습한 CNN은 모양(Shape)을 잘 구분하지 못하고 텍스쳐(Texture)에 편향되어 있다고 알려져 있는데요[14]. 이에 따라 ImageNet Sketch 데이터셋에서는 기존 ResNet[2] 모델은 형편 없는 성능을 보여주고 있죠. 반면 CLIP 모델은 기존 ImageNet 성능에서 크게 떨어지지 않는 성능을 보여주고 있습니다. 마찬가지로 Adversarial Attack이 추가된 ImageNet-A 데이터셋에서도 기존 ResNet[2]은 성능이 크게 하락한 반면, CLIP 모델은 다른 데이터셋에서와 비슷한 성능을 보여주죠.

이러한 성능은 기존 Vision Model의 특성과 비교해봤을 때 분명 혁신적인 수준입니다. 단순히 성능이 좋다의 문제가 아니라, 기존 Vision Model의 한계를 극복할 수 있는 하나의 가능성을 보여준 것이죠. 이러한 생각은 CLIP이 발표되었던 2021년 당시보다 다양한 Multimodal Model들이 발표되고 있는 2023년 현재 더욱 뚜렷해보입니다. 이미지-자연어 쌍을 학습한 Multimodal Model들은 기존 모델에서는 상상할 수 없는 다양한 기능들을 선보이며 Large Multimodal Model의 전성기를 열어내고 있습다.

## **5. 토론**

지금까지 CLIP 이전 모델들의 한계는 무엇인지, 그리고 CLIP은 이러한 한계를 어떻게 극복했는지 살펴봤습니다. 또한 다양한 실험 결과를 통해 실제로 기존 방법의 한계를 극복했음을 확인했습니다.

이번 챕터에서는 CLIP 논문에서 제기하고 있는 토론 내용을 살펴보겠습니다. CLIP 논문에서는 한가지 중요한 질문을 던지고 있는데요. 바로 사람과의 비교입니다. ImageNet 분류 문제에서 인간의 점수를 넘어선 뒤로 Vision Model에서는 사람과의 비교를 하는 내용은 좀처럼 찾아볼 수 없었는데요. 흥미롭게도 CLIP에서는 CLIP의 Zero Shot 기능과 인간의 Few Shot 기능에 대해 비교하고 있습니다.

![https://ffighting.net/wp-content/uploads/2023/11/image-9.png](https://ffighting.net/wp-content/uploads/2023/11/image-9.png)

그림10. 사람과의 비교

위 그림은 사람과 CLIP의 Zero Shot, One Shot, Two Shot 성능을 비교한 표 입니다. Accuracy 성능만 놓고 비교해볼게요. 우선 CLIP의 Zero Shot 성능은 인간의 Zero Shot, One Shot 심지어 Two Shot 성능 보다도 높습니다. 하지만 저자들이 주목하는건 One Shot 성능입니다. 사람은 Zero Shot 성능은 크게 떨어지지만, 하나의 샘플을 참고하고 나면 크게 점수가 오르는데요. 재밌는건 두개의 샘플을 본다고 해서 성능이 더 올라가지도 않습니다. 반면 CLIP은 Zero Shot 성능이 좋지만, 하나의 샘플을 사용하여 재학습 했을때 오히려 성능이 하락하는 모습을 보이죠.

이러한 CLIP의 특성을 우선 변호해줄 필요도 있는데요. 왜냐하면 Zero Shot 성능에 특화되도록 설계된 CLIP이 그동안의 학습 과정을 무시하고 Few shot으로 재학습하는 과정에서 기존 학습된 파라미터가 모두 변경되면서 성능이 하락한 것으로 볼 수 있기 때문이죠. 그렇긴 하지만 저자들이 주목하는 부분은 바로 인간의 특성입니다. 인간은 하나의 샘플만 보아도 크게 성적이 올라가죠. 생각해보면 우리는 무엇을 알고 있는지, 무엇을 모르고 있는지를 알고 있습니다. 따라서 단 하나의 샘플마 주어졌어도 기존 나의 지식과 비교해가며 새로운 정보를 잘 일반화하여 해석하죠. 이러한 사람의 특징을 메타 인지라고 하는데요. 저자들은 CLIP에는 이러한 부분이 부족함을 강조합니다. 즉 아직은 사람 뇌의 기능을 따라가려면 멀었다는 것이죠.

이렇게 아직 사람 뇌와는 기능적인 차이가 분명 존재함에도 불구하고 CLIP 모델은 분명한 특장점을 갖습니다. 바로 기존 Vision Model의 한계를 극복했다는 점인데요. 앞서 여러번 강조했듯이 CLIP은 기존 방법들과 달리 이미지-자연어 쌍 데이터를 학습했죠. 그 덕분에 기존 모델에서는 부족했던 일반화 성능과 강건화 성능을 올릴 수 있었습니다.

## **6. 장단점**

지금까지 CLIP 논문의 핵심 내용을 살펴봤습니다. 지금까지의 내용을 바탕으로 CLIP 논문의 장단점을 정리해보겠습니다.

### **6-1. 장점**

첫 번째 장점은 다양한 이미지와 텍스트 데이터에 대한 강력한 이해력입니다. CLIP은 대규모의 이미지와 텍스트 데이터를 통해 학습되며, 이를 통해 다양한 유형의 이미지와 관련된 텍스트를 이해하는 능력을 갖춥니다. 이는 CLIP이 다양한 시나리오와 주제에 대해 높은 수준의 이해력을 보여준다는 것을 의미합니다.

두 번째 장점은 Zero Shot 학습 능력 입니다. CLIP은 제로샷 학습을 통해 본 적 없는 새로운 데이터에 대해서도 분류 및 인식 작업을 수행할 수 있습니다. 이는 모델이 새로운 상황에 빠르게 적응하고, 추가적인 학습 데이터 없이도 효과적으로 작동할 수 있음을 의미합니다.

세 번째 장점은 자연어 처리 모델과 이미지 인식 모델의 결합입니다. CLIP은 이미지 인식과 자연어 처리를 결합하여, 이미지에 대한 설명을 자연어로 생성하거나, 반대로 텍스트를 통해 이미지를 분류하는 등의 작업을 수행할 수 있습니다. 이는 두 분야의 강점을 결합하여 보다 풍부하고 다양한 응용을 가능하게 합니다.

네 번째 장점은 대규모 데이터셋을 효율적으로 학습할 수 있는 방법을 제안했다는 점입니다. CLIP은 대규모 데이터셋에서 효율적으로 학습됩니다. 이는 모델이 방대한 양의 데이터에서 복잡한 패턴과 관계를 학습할 수 있음을 의미하며, 이를 통해 더 정확하고 신뢰할 수 있는 결과를 도출할 수 있습니다.

다섯 번째 장점은 다양한 태스크에 대해 유연함을 보인다는 점입니다. CLIP은 이미지 분류, 객체 탐지, 세그멘테이션 등 다양한 컴퓨터 비전 태스크에 적용될 수 있습니다. 이는 모델이 다양한 문제를 해결하는 데 사용될 수 있음을 의미하며, 이를 통해 다양한 응용 분야에서의 활용 가능성이 높아집니다.

이러한 장점들은 CLIP을 다양한 분야에서 활용할 수 있는 강력한 도구로 만들어줍니다. 이미지와 텍스트 데이터를 결합한 학습 방식은 특히 이미지와 관련된 텍스트 정보를 처리하는 데 있어 매우 효과적입니다.

### **6-2. 단점**

첫 번째 단점은 제한된 Zero Shot 학습 성능입니다. CLIP은 제로샷 학습을 지원하지만, 이러한 학습 방식은 아직 전통적인 학습 방식에 비해 성능이 떨어질 수 있습니다. 특히, 복잡하거나 세부적인 분류 작업에서는 제로샷 학습의 한계가 드러날 수 있습니다.

두 번째 단점은 대규모 데이터셋에 대한 의존성입니다. CLIP의 효과적인 학습을 위해서는 대규모의 이미지와 텍스트 데이터셋이 필요합니다. 이는 데이터 수집과 처리에 많은 시간과 자원이 필요하며, 모든 경우에 이러한 대규모 데이터셋을 확보하는 것이 가능하지 않을 수 있습니다.

세 번째 단점은 계산 비용 및 자원 소모입니다. CLIP의 학습과 추론 과정은 상당한 계산 자원을 요구합니다. 이는 고성능의 하드웨어가 필요하며, 특히 대규모 모델의 경우 비용이 매우 높을 수 있습니다.

네 번째 단점은 아직 특정 분야에서는 일반화가 어렵다는 점입니다. CLIP은 다양한 데이터셋에 대해 학습되지만, 특정 유형의 데이터에 대해서는 일반화하는 데 어려움을 겪을 수 있습니다. 특히, 학습 데이터셋에서 충분히 대표되지 않은 유형의 데이터에 대해서는 성능이 저하될 수 있습니다.

다섯 번째 단점은 편향입니다. 인터넷에서 수집된 이미지와 텍스트 데이터를 사용하는 CLIP은 데이터에 내재된 사회적 편향을 학습할 위험이 있습니다. 이는 모델이 편향된 결과를 생성할 수 있으며, 이러한 편향을 식별하고 수정하는 것은 매우 어려운 작업입니다.

여섯 번째 단점은 복잡한 태스크에 대한 한계입니다. CLIP은 기본적인 이미지 분류나 객체 인식과 같은 태스크에는 효과적이지만, 더 복잡하고 추상적인 태스크를 수행하는 데는 한계가 있습니다. 예를 들어, 이미지 내 객체의 수를 세는 등의 시스템적인 태스크에서는 성능이 떨어질 수 있습니다.

이러한 단점들은 CLIP 모델을 사용할 때 고려해야 할 중요한 요소들입니다. 특히, 대규모 데이터셋에 대한 의존성과 계산 비용은 실제 응용 환경에서의 사용을 제한할 수 있는 주요 요인입니다. 또한, 사회적 편향과 일반화의 어려움은 모델의 신뢰성과 공정성에 영향을 미칠 수 있습니다.

## **7. 의의**

첫 번째 의의는 자연어와 이미지의 의미있는 융합 방법을 제안한 것입니다. CLIP은 자연어 처리와 컴퓨터 비전을 결합하는 혁신적인 접근 방식을 제시합니다. 이 모델은 텍스트와 이미지 사이의 관계를 이해하고, 이를 통해 이미지를 더욱 풍부하고 정확하게 분석할 수 있습니다. 이는 인공지능이 인간의 언어와 시각적 세계를 더욱 잘 이해할 수 있게 하는 중요한 발전입니다.

두 번째 의의는 Zero Shot 학습 방법의 가능성을 보여준 것입니다. CLIP은 제로샷 학습, 즉 사전에 본 적 없는 데이터에 대해서도 분류 및 인식이 가능한 모델을 제공합니다. 이는 기존에는 불가능했던 새로운 유형의 문제 해결 방법을 제시하며, 더 적은 데이터로도 효과적인 학습이 가능하게 합니다.

세 번째 의의는 다양한 데이터셋에 대한 적용성을 보여준 것입니다. CLIP은 다양한 유형의 이미지와 텍스트 데이터셋에 적용 가능합니다. 이는 모델이 다양한 환경과 상황에서 유연하게 사용될 수 있음을 의미하며, 다양한 분야에서의 응용 가능성을 열어줍니다.

네 번째 의의는 편향에 대한 인식과 대응을 촉발했다는 점입니다. CLIP은 데이터에 내재된 사회적 편향을 학습할 수 있는 위험을 인식하고, 이에 대응하기 위한 연구의 필요성을 강조합니다. 이는 인공지능 기술의 공정성과 윤리적 사용에 대한 중요한 논의를 촉진시키는 계기가 됩니다.

다섯 번째는 인공지능 연구의 새로운 방향을 제시했다는 점입니다. CLIP은 인공지능 연구에 있어 새로운 방향을 제시합니다. 특히, 다양한 유형의 데이터를 통합적으로 학습하는 방식은 향후 인공지능 모델의 발전에 중요한 영향을 미칠 것으로 예상됩니다.

여섯 번째 의의는 실용적인 응용 가능성을 제시했다는 점입니다. CLIP의 접근 방식은 실제 세계의 문제 해결에 적용될 수 있는 실용적인 가능성을 보여줍니다. 예를 들어, 소셜 미디어에서의 이미지와 텍스트 분석, 의료 이미지의 해석, 또는 디지털 아카이브에서의 자료 검색 등 다양한 분야에서 활용될 수 있습니다.

CLIP 모델의 이러한 의의는 인공지능 기술의 발전뿐만 아니라, 사회적, 윤리적 측면에서도 중요한 의미를 가집니다. 이는 인공지능이 인간의 언어와 시각을 더욱 깊이 이해하고, 다양한 분야에서 유용하게 활용될 수 있는 길을 열어주는 중요한 발전입니다.

## **8. 마치며**

이 글을 통해 CLIP 논문의 주요 내용과 그 의미에 대해 함께 살펴보았습니다. CLIP은 기존의 컴퓨터 비전 모델과 언어 모델의 한계를 넘어서려는 중요한 시도로, 이 분야의 발전에 큰 기여를 하고 있습니다. 이 글을 통해 여러분이 CLIP의 중요성과 그 가능성을 깊이 이해하셨기를 바랍니다.

우리는 CLIP 논문을 통해 기존 방법의 문제점을 이해하고, CLIP이 제안하는 새로운 접근 방식을 살펴보았습니다. 대규모 이미지-자연어 쌍 데이터셋을 활용한 학습 방법과 zero shot prediction의 개념은 이 분야에서의 새로운 발전을 이끌고 있습니다. 또한, 실험 결과를 통해 CLIP 모델의 성능과 강건함을 확인할 수 있었습니다.

토론 부분에서는 사람과 CLIP의 차이점을 탐구하며, 이러한 차이가 모델의 성능과 학습 방식에 어떤 영향을 미치는지 이해할 수 있었습니다. 또한, CLIP 모델의 장점과 단점을 구체적으로 분석하며, 이 모델이 가진 한계와 발전 가능성을 탐색했습니다. 마지막으로, CLIP의 의의를 살펴보며, 이 모델이 인공지능 분야에 어떤 새로운 방향성을 제시하는지 고찰했습니다.

이 글을 마무리하며, CLIP 논문이 제시하는 새로운 방향성이 인공지능 분야의 미래에 어떤 영향을 미칠지 기대해보고자 합니다. CLIP은 단순히 새로운 기술을 넘어서, 인공지능 연구와 개발의 새로운 지평을 열고 있습니다. 이 글이 여러분에게 CLIP 논문의 중요성을 이해하는 데 도움이 되었기를 바라며, 앞으로도 이 분야의 발전을 지켜보는 것이 매우 흥미로울 것입니다. 여러분의 지속적인 관심과 통찰이 이 분야의 발전에 큰 도움이 될 것입니다.

## **8. 참고자료**

1. [Inception 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/inception/)
2. [ResNet 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/resnet/)
3. [SENet 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/senet/)
4. [BAM 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/bam/)
5. [CBAM 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/cbam/)
6. [ImageGPT 논문](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)
7. [Vision Transformer 논문 리뷰](https://ffighting.net/deep-learning-paper-review/vision-model/vision-transformer/)
8. [Transformer 논문 리뷰](https://ffighting.net/deep-learning-paper-review/language-model/transformer/)
9. [GPT-1 논문 리뷰](https://ffighting.net/deep-learning-paper-review/language-model/gpt-1/)
10. [GPT-2 논문](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
11. [GPT-3 논문 리뷰](https://ffighting.net/deep-learning-paper-review/language-model/gpt-3/)
12. [BERT 논문 리뷰](https://ffighting.net/deep-learning-paper-review/language-model/bert/)
13. [SimCLR 논문 리뷰](https://ffighting.net/deep-learning-paper-review/self-supervised-learning/simclr/)
14. [[ICLR 2019] IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS](https://arxiv.org/pdf/1811.12231.pdf)

[CLIP 논문 리뷰 - Learning Transferable Visual Models From Natural Language Supervision](https://ffighting.net/deep-learning-paper-review/multimodal-model/clip/)

---

---

결론 : 

[https://greeksharifa.github.io/computer vision/2021/12/19/CLIP/](https://greeksharifa.github.io/computer%20vision/2021/12/19/CLIP/)