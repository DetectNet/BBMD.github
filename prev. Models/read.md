**기존에 있었던 연구들을 리뷰(본 연구에서 새로 제작한 모델들과 성능을 비교하기 위해 제작)**

* XGBClassifier_based_on_opcode_frequency

-> 기존 Malware Detection Machine learning 모델들 중 Opcode Frequency(한 PEFILE 내에서 특정 Opcode가 몇번 발생했는지의 여부)를 입력을 통해 악성코드를 탐지하는 XGBClassifier 기반 탐지 모델

* XGBClassifier_based_on_word2vecEmbedding

-> PEFILE내의 Opcode Sequence List를 바탕으로 word2vec 모델에 opcode에 대한 임베딩값을 얻기 위한 학습을 진행하고, 학습시킨 word2vec모델로 pefile의 opcode Sequence List의 임베딩 값으로 XGBClassifier을 학습시켜, Malware을 탐지하는 모델

* file

-> 기존 모델들을 작동시키기 위해 필요한 기본적인 데이터들이 저장된 디렉토리

* utils

-> 각 기존 머신러닝 기반 탐지모델을 위한 util 파일 및 sample 파일이 저장된 디렉토리
