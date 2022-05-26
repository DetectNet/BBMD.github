# :jack_o_lantern: BBMD 악성코드 탐지 프로젝트
*NLP Term Project: BPE BASED MALWARE DETECTECTION*
> "BPE 알고리즘을 이용하여 opcode의 순서를 반영하면 악성코드 탐색 효율을 높일수 있지 않을까?"

라는 궁금증에서 시작한 프로젝트입니다. 기존 악성코드 분석 방법에서는 추출된 opcode들의 순서를 신경 쓰지 않았다는 특징을 발견하였습니다.

프로젝트의 최종 목표는 아래와 같습니다.

**1. BPE를 활용하여 단순히 opcode frequency 또는 임베딩을 한 기존 기법보다 더 발전된 탐지 모델 제작**

**2. OPCODE 의 순서를 반영하지 않고 측정한 기존 악성코드 탐지의 정확도와 자체적으로 튜닝한 BPE를 적용하여 정확도를 비교** 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_(Sequence를 활용하는 추가적인 모델 활용 예정 ex) seq2seq, attention, LSTM, transformer 등)_

**3. 기존 악성 코드 탐지 기법 중, 전처리 과정에서의 새로운 방법론 제안** 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_(BPE를 통해 생성한 token의 불용어 추가 예정)_

## :apple: 기대 효과
**1. 기존 분석모델의 효율성 증대**
> 기존에는 Opcode 리스트가 굉장히 길다는 이유로 opcode의 순서를 반영하지 않았습니다. NLP 분야의 BPE Tokenizer 알고리즘을 통해 순서를 반영하여 기존 모델의 정확도를 높히는 결과를 기대합니다
 
**2. 적용 범위의 확장**
> NLP 분야에서 활용되는 전처리 과정이 악성코드 분석 분야에서 유용한지를 확인합니다. 이를 통해 범용성이 입증 된다면, NLP 전처리 과정이 다양한 분야에서도 확장하여 응용이 가능하다는 가능성을 입증합니다.

## Working Enviornment
* Google Colab
* Python Version : 3.7.13
