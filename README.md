# :jack_o_lantern: BBMD 악성코드 탐지 프로젝트
*NLP Term Project: BPE BASED MALWARE DETECTECTION*
> "BPE 알고리즘을 이용하여 opcode의 순서를 반영하면 악성코드 탐색 효율을 높일수 있지 않을까?"

라는 궁금증에서 시작한 프로젝트입니다. 기존 악성코드 분석 방법에서는 추출된 opcode들의 순서를 신경 쓰지 않았다는 특징을 발견하였습니다.

프로젝트의 최종 목표는 아래와 같습니다.

**1. BPE를 활용하여 단순히 opcode frequency 또는 임베딩을 한 기존 기법보다 더 발전된 탐지 모델 제작**

**2. OPCODE 의 순서를 반영하지 않고 측정한 기존 악성코드 탐지의 정확도와 자체적으로 튜닝한 BPE를 적용하여 정확도를 비교** 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_(Sequence를 활용하는 추가적인 모델 활용 예정 ex) seq2seq, attention, LSTM, transformer 등)_

**3. 기존 악성 코드 탐지 기법 중, 전처리 과정에서의 새로운 방법론 제안** 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_(BPE를 통해 긴 하나의 pefile의 opcode List를 압축하는 방식 제안)_

## :apple: 기대 효과
**1. 기존 분석모델의 효율성 증대**
> 기존에는 Opcode 리스트가 굉장히 길다는 이유로 opcode의 순서를 반영하지 않았습니다. NLP 분야의 BPE Tokenizer 알고리즘을 통해 순서를 반영하여 기존 모델의 정확도를 높히는 결과를 기대합니다
 
**2. 적용 범위의 확장**
> NLP 분야에서 활용되는 전처리 과정이 악성코드 분석 분야에서 유용한지를 확인합니다. 이를 통해 범용성이 입증 된다면, NLP 전처리 과정이 다양한 분야에서도 확장하여 응용이 가능하다는 가능성을 입증합니다.

## 🌲Working Enviornment
* Google Colab
* Python Version : 3.7.13




## 🌟 결과
![KakaoTalk_20220527_170538757](https://user-images.githubusercontent.com/101659578/170662635-93601d23-33ab-45d5-b234-be2d22ff17ed.png)

결과 테이블
![그림1](https://user-images.githubusercontent.com/101659578/170675642-b1baf705-7ee5-4d81-8dd2-6e830ae0dd6b.png)




`👉 각 모델의 파일 설명은 디렉토리에 있는 read file에 기재하였, 동작 과정은 각각의 ipynb의 text로 설명하였습니다`


## 🔎 추가 진행 계획 및 개선 사항
* _OPCODE_ 중 _nop_ 가 아무 의미가 없는 버퍼형식으로 작동하는것을 발견하였습니다. _nop_ 를 제거하는 방식으로 성능을 높일 수 있다 판단하여, 프로젝트를 발전시켜볼 계획입니다.
* Sequence 데이터를 활용할 경우, 굉장히 긴 데이터를 모델에 Input을 더 잘반영하는 모델을 제작하기 위해 긴 데이터를 줄이는 작업이 필요하고, 악성코드 BPE 토큰들 중 불용어를 선택하기 위해 TF-IDF를 통해 값이 0인 토큰들을 불용어로 가정하는 작업들을 진행해볼 예정입니다.
* Sequence를 학습하는 모델이 뒤의 리스트를 반영하지 못한다는 단점을 해결하기 위해 추가적인 모델 활용 예정(ex) seq2seq, attention, transformer 등)입니다. 본 연구에서는 3000을 max_length로 설정하여 LSTM+CNN 모델을 제작하였을 경우에도 99%의 정확도를 확인하였지만, 뒤의 opcode를 반영하지 못하고 slice 되는 문제를 해결하기 위해 seq2seq등을 통해 뒤 리스트를 예측하도록 인코더와 디코더를 학습시킨 뒤, 인코더에서 추출한 Feature로 학습하는 모델을 제작할 예정입니다.
* TBPTT기반 순환신경망 모델을 제작 예정입니다. TBPTT는 순환신경망의 전체적인 가중치를 학습하는 과정속에서 발생되는 Gradient Descent를 방지하기 위해 사용되는 오차역전파 방식입니다. TBPTT는 자연어처리 분야에서 Performance 관점에서 뛰어난 성능을 내지 못해 잘 사용되지는 않으나, 긴 시계열 데이터를 학습하는 과정속에서 기울기 소실문제를 완전히 방지한다는 장점을 가지고 있고 본 연구에서는 이를 활용하여 성능을 측정할 예정입니다.
* 기존 모델들의 경우, 난독화가 이루어진 악성코드를 잘 예측하지 못한다는 단점이 존재합니다. 본 프로젝트는 Sequence를 입력 받는 모델의 성능이 기존 모델들과 비슷한 성능을 내는 것을 확인하였고, 이를 바탕으로 난독화된 code와 원본 code 사이의 loss를 줄이는 consistency Training과 BERT와 같이 사전 학습을 하되, 난독화와 원본 사이의 관계를 예측하게 하는 등의 Custom Pre-train 방식을 개발하여 단점을 극복할 예정입니다.
