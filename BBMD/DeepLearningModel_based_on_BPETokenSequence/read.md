**opcode sequence를 입력받는 딥러닝 모델을 제작하기 위해 opcode sequence를 BPE Token sequence로 변환을 한 뒤, LSTM+CNN모델의 Input data로 학습을 하고 테스트를 한 모델**

* BPETokenSequenceBasedLSTMCNNModel.ipynb

> opcode sequence를 입력받는 딥러닝 모델을 제작하기 위해 opcode sequence를 BPE Token sequence로 변환을 한 뒤, LSTM+CNN모델의 Input data로 학습을 하고 테스트를 한 모델

> 학습 결과 모델 구조에 가중치 값인 BPESequenceBasedLSTMCNNMalwareDetectionModel.h5를 제작하였다.

* BPESequenceBasedLSTMCNNMalwareDetectionModel.zip
* BPESequenceBasedLSTMCNNMalwareDetectionModel.z01
* BPESequenceBasedLSTMCNNMalwareDetectionModel.z02
* BPESequenceBasedLSTMCNNMalwareDetectionModel.z03

> 학습을 통해 생성된 모델의 가중치인 BPESequenceBasedLSTMCNNMalwareDetectionModel.h5 파일이 Github에서 제공한 
25MB보다 용량이 커서 분할 압축을 통해 업로드가 되어진 파일

> 압축 해제하시고 test.ipynb 파일에서 사용하시는 것을 추천 드립니다.

* test.ipynb

> 위의 모델을 활용해서 악성코드 샘플 파일 및 양성코드 샘플파일을 탐지하는 예시를 보여주는 파일입니다.
