**각 기존 머신러닝 기반 탐지모델을 위한 util 파일 및 sample 파일 저장**

* 3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe

> 양성 파일 샘플

* MalwareSample.p

> 악성코드 파일을 직접 올리기 어려워 opcode sequence만 미리 추출하여 pickle 형태로 저장한 파일

* OpcodeExtracter.ipynb

> 본 연구에서 먼저 대량의 Pefile을 추출하기 위해 사용한 코드

* test.ipynb

> utils.py 내부의 함수들이 잘 작동하는지 테스트하기 위해 사용한 코드

* utils.py

> 각 모델들이 작동 및 opcode 추출, 변환 등 다양한 util 함수를 미리 정의한 모듈 코드
