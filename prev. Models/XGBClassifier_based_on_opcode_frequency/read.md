**기존 Malware Detection Machine learning 모델들 중 Opcode Frequency(한 PEFILE 내에서 특정 Opcode가 몇번 발생했는지의 여부)를 입력을 통해 악성코드를 탐지하는 XGBClassifier 기반 탐지 모델**

* OpcodeFrequecnyBasedXGBClassifierMalwareDetectionModel.ipynb
-> Opcode빈도수를 활용해서 악성코드를 분류하는 XGBClassifier 기반 분류 모델을 제작하는 코드
-> 분류 모델 제작 과정 및 정확도를 확인할 수 있습니다.

* XGBClassifier.model
-> OpcodeFrequecnyBasedXGBClassifierMalwareDetectionModel.ipynb를 통해 생성된 악성코드 탐지 분류 모델 파일

* test.ipynb
-> 악성코드 탐지 모델을 바탕으로 양성파일 샘플을 잘 분류했는지 확인해주는 코드
-> 악성코드 파일은 샘플을 올리지 못하는 관계로 opcode sequence를 pickle 형태로 저장된 파일을 통해 확인하면 됩니다.
