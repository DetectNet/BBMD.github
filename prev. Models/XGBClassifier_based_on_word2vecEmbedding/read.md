PEFILE내의 Opcode Sequence List를 바탕으로 word2vec 모델에 opcode에 대한 임베딩값을 얻기 위한 학습을 진행하고, 
학습시킨 word2vec모델로 pefile의 opcode Sequence List의 임베딩 값으로 XGBClassifier을 학습시켜, Malware을 탐지하는 모델
