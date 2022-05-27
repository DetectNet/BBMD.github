**제작한 모델들**

* DeepLearningModel_based_on_BPETokenSequence

> opcode sequence를 입력받는 딥러닝 모델을 제작하기 위해 opcode sequence를 BPE Token sequence로 변환을 한 뒤, LSTM+CNN모델의 Input data로 학습을 하고 테스트를 한 모델

* XGBClassifier_based_on_BPE_Token_word2vecEmbedding/

> BPE Token을 2000개 만들었을 경우, 각 토큰들에 대한 word2vec 임베딩 학습을 시키고, 추후 임베딩을 하여 해당 값을 기반으로 XGBClassifier분류 모델을 통해 학습을 하고 악성코드를 탐지하는 모델

* file

> 모델들을 작동시키기 위해 필요한 기본적인 데이터들이 저장된 디렉토리

* utils

> 머신러닝 기반 탐지모델을 위한 util 파일 및 sample 파일이 저장된 디렉토리
