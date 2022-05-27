**BPE Token을 2000개 만들었을 경우, 각 토큰들에 대한 word2vec 임베딩 학습을 시키고, 추후 임베딩을 하여 해당 값을 기반으로 XGBClassifier분류 모델을 통해 학습을 하고 악성코드를 탐지하는 모델**
- 해당 방식은 기존 연구 방식 중 가장 성능이 좋은 word2vec 임베딩 값을 통해 분류하는 방식보다 더 좋은 성능을 내었다.
- 이를 통해 본 연구에서 활용할 BPE Token이 잘 생성되고 있음을 확인할 수 있었다.

* BPE2000 word2vec + XGBoost Classifier.ipynb

> BPE Token sequence 정보를 기반으로 각 BPE Token들에 대해 word2vec 임베딩 값이 주어지도록 학습을 진행하고, 학습시킨 word2vec 모델을 기반으로 Sequence를 임베딩하여 XGBClassifier 기반 분류모델을 통해 악성코드를 탐지하도록 모델을 학습시키는 코드

>> 이 코드를 통해 BPE Token을 word2vec 임베딩 값을 제공하는 BPEw2v2000 모델과, 임베딩 값을 기반으로 예측하는 탐지모델인 XGB_BPE2000_Word2vecClassifier.model이 생성됩니다.

* BPEw2v2000

> BPE Token sequence를 통해 학습된 BPE Token에 대한 word2vec 모델입니다.

* XGB_BPE2000_Word2vecClassifier.model

> word2vec 임베딩 값을 기반으로 악성코드를 탐지하는 모델입니다.

* test.ipynb

> 위의 두 모델을 활용해서 악성코드 샘플 파일 및 양성코드 샘플파일을 탐지하는 예시를 보여주는 파일입니다.
