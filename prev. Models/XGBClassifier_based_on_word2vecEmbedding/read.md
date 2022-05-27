**PEFILE내의 Opcode Sequence List를 바탕으로 word2vec 모델에 opcode에 대한 임베딩값을 얻기 위한 학습을 진행하고, 
학습시킨 word2vec모델로 pefile의 opcode Sequence List의 임베딩 값으로 XGBClassifier을 학습시켜, Malware을 탐지하는 모델**

* word2vec + XGBoost Classifier.ipynb

-> opcode sequence 정보를 기반으로 각 opcode들에 대해 word2vec 임베딩 값이 주어지도록 학습을 진행하고, 학습시킨 word2vec 모델을 기반으로 Sequence를 임베딩하여 XGBClassifier 기반 분류모델을 통해 악성코드를 탐지하도록 모델을 학습시키는 코드

-> 이 코드를 통해 opcode를 word2vec 임베딩 값을 제공하는 w2v 모델과, 임베딩 값을 기반으로 예측하는 탐지모델인 XGB_Word2vecClassifier.model이 생성됩니다.

* w2v

-> opcode sequence를 통해 학습된 opcode에 대한 word2vec 모델입니다.

* XGB_Word2vecClassifier.model

-> word2vec 임베딩 값을 기반으로 악성코드를 탐지하는 모델입니다.

* test.ipynb

-> 위의 두 모델을 활용해서 악성코드 샘플 파일 및 양성코드 샘플파일을 탐지하는 예시를 보여주는 파일입니다.
