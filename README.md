# :jack_o_lantern: BBMD 악성코드 탐지 프로젝트
*NLP Term Project: BPE BASED MALWARE DETECTECTION(BBMD)*
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
* numpy==1.21.6
* pandas==1.3.5
* tqdm==4.64.0
* capstone==4.0.2
* pefile==2021.9.3
* tensorflow==2.8.0
* keras==2.8.0
* xgboost==0.90
* gensim==3.6.0
* sklearn==1.0.2

## 🗃️Dataset
* DikeDataset: https://github.com/iosifache/DikeDataset

## 🗂️각 디렉토리별 설명
* prev.Models: 기존에 연구된 기법들을 재현한 모델들을 시현해볼 수 있습니다.(opcode Frequency 기반, word2vec 임베딩 기반)
* BBMD: BPE 토큰을 사용해서 새로 연구된 기법을 바탕으로 제작된 모델을 시현해볼 수 있습니다.(BPE Token word2vec 임베딩 기반, BPE Token Sequence 입력 기반)
* 각 모델들의 디렉토리에서 ipynb 확장자 파일들을 보시면 모델 구조 및 학습을 어떻게 하였는지를 확인하실 수 있고, test.ipynb 파일을 확인하시면 각 모델을 어떻게 사용하는지 설명을 확인하실 수 있습니다.

## 📃How to use
* 사용법은 각 디렉토리에 있는 test.ipynb를 보시면 쉽게 이해하실 수 있습니다.
* BBMD.github에서 requirements.txt 내 패키지들이 전부 설치되어야 작동됩니다.

### utils.py 파일 사용법
* BBMD 안에 있는 utils 디렉토리에 있는 utils.py를 사용하는 방법입니다.
* utils 디렉토리 내부에 있는 test.ipynb 파일을 쥬피터 노트북을 통해 읽으시면 쉽게 이해하실 수 있습니다.
```python
>>> # 필요한 패키지인 pefile과 capstone이 설치된 환경에서 작동됩니다.
>>> # utils.py가 있는 파일로 디렉토리를 이동하시고 사용하여야 합니다.
>>> from utils import *
>>> # opcodeList가 있는 파일과 BPE Token vocab 파일을 로드를 해줍니다.
>>> with open('opcodesList.txt', 'rb') as lf:
>>>     opcodes = pickle.load(lf)
>>> opcodes2 = []
>>> for i in range(len(opcodes)):
>>>    opcodes2.append([opcodes[i]])
>>> with open('2000vocab.p', 'rb') as file:
>>>    vocab = pickle.load(file)
>>> vocab = vocab + opcodes2
>>> # utils 내 함수 동작 방법들입니다.
>>> # 특징이 추출될 pefile 이름 및 불러오기
>>> fileName = '3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe'
>>> try:
>>>    exe = pefile.PE(fileName)
>>> except:
>>>    print('Error File')
>>> # ExtractPefileOpcodes: pefile 내부의 opcode를 전체 추출(순서대로)
>>> sampleOpcodeList = ExtractPefile12000Opcodes(opcodes, exe)
>>> print(sampleOpcodeList[:10])
['add', 'add', 'add', 'add', 'dec', 'mov', 'xor', 'dec', 'lea', 'inc']
>>> # Tokenizer: opcode sequence list를 BPE token list로 변환
>>> sampleTokens = Tokenizer(vocab, sampleOpcodeList)
>>> print(sampleTokens[:10])
['addaddaddadd', 'decmovxor', 'decleaincmov', 'inc', 'movdec', 'cmpinc', 'movincinc', 'testjs', 'incmov', 'dectestje']
>>> # TokenIdMapping: BPE Token list를 BPE Token Id로 변환
>>> sampleTokenIdList = TokenIdMapping(vocab, sampleTokens)
>>> print(sampleTokenIdList[:10])
[2110, 1336, 606, 86, 2045, 1837, 709, 1499, 2059, 1851]
```
### prev.Models/XGBClassifier_based_on_opcode_frequency/모델 사용법
* 디렉토리 내부에 있는 test.ipynb 파일을 쥬피터 노트북을 통해 읽으시면 쉽게 이해하실 수 있습니다.
```python
>>> # 필요한 패키지인 pefile과 capstone이 설치된 환경에서 작동됩니다.
>>> # utils.py가 있는 파일로 디렉토리를 이동하시고 사용하여야 합니다.
>>> # 필요한 패키지 import
>>> from utils import *
>>> from xgboost import XGBClassifier
>>> # opcode list 로드(utils 디렉토리 내부에 있는 opcodeList.txt를 로드하시면 됩니다.)
>>> with open('opcodesList.txt', 'rb') as lf:
>>>     opcodes = pickle.load(lf)
>>> opcodeDict = {}
>>> for i in range(len(opcodes)):
>>>     opcodeDict.setdefault(opcodes[i], i)
>>> # 학습된 모델을 로드해줍니다.
>>> # 파일명
>>> filename = 'XGBClassifier.model'
>>> # 모델 불러오기
>>> clf = pickle.load(open(filename, 'rb'))
>>> # test.ipynb 파일 내부에 있는 MalwareDetectionFunction 함수와 MalwareDetectionFunctionUsingPickle 함수를 만들어주세요(길어서 여기서는 생략합니다.)
>>> # 양성 파일 테스트(0으로 출력될 경우 양성)
>>> fileName = '3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe'
>>> MalwareDetectionFunction(clf, fileName)
0
>>> # 악성코드 파일 테스트(1로 출력될 경우 악성)
>>> # 악성코드 파일을 직접 Google Drive 및 GitHub에 올릴 수 없기 때문에 pickle로 먼저 opcodeSequence를 추출하여 해당 파일을 바탕으로 악성코드를 탐지하는 함수를 제작하였습니다.
>>> pickleName = 'MalwareSample.p' 
>>> MalwareDetectionFunctionUsingPickle(clf, pickleName)
1
```
### prev.Models/XGBClassifier_based_on_word2vecEmbedding/모델 사용법
* 디렉토리 내부에 있는 test.ipynb 파일을 쥬피터 노트북을 통해 읽으시면 쉽게 이해하실 수 있습니다.
```python
>>> # 필요한 패키지인 pefile과 capstone이 설치된 환경에서 작동됩니다.
>>> # utils.py가 있는 파일로 디렉토리를 이동하시고 사용하여야 합니다.
>>> # 필요한 패키지 import
>>> from utils import *
>>> from xgboost import XGBClassifier
>>> from gensim.models import Word2Vec
>>> from gensim.models import KeyedVectors
>>> # opcode list 로드(utils 디렉토리 내부에 있는 opcodeList.txt를 로드하시면 됩니다.)
>>> with open('opcodesList.txt', 'rb') as lf:
>>>     opcodes = pickle.load(lf)
>>> # 학습된 XGBClassifier 및 word2vec 모델 로드
>>> # 파일명
>>> filename = 'XGB_Word2vecClassifier.model'
>>> # 모델 불러오기
>>> clf = pickle.load(open(filename, 'rb'))
>>> # word2vec 모델 로드
>>> modelPath = 'w2v'
>>> word2vecModel = KeyedVectors.load_word2vec_format(modelPath)
>>> # test.ipynb 파일 내부의 get_sentence_mean_vector(morphs) 함수를 구현하셔서 pefile을 word2vec 임베딩 값으로 변경해주는 함수를 만들어주셔야 합니다.(길어서 여기서는 생략합니다.)
>>> # test.ipynb 파일 내부에 있는 MalwareDetectionFunction 함수와 MalwareDetectionFunctionUsingPickle 함수를 만들어주세요(길어서 여기서는 생략합니다.)
>>> # 양성 파일 테스트(0으로 출력될 경우 양성)
>>> fileName = '3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe'
>>> MalwareDetectionFunction(clf, fileName)
0
>>> # 악성코드 파일 테스트(1로 출력될 경우 악성)
>>> # 악성코드 파일을 직접 Google Drive 및 GitHub에 올릴 수 없기 때문에 pickle로 먼저 opcodeSequence를 추출하여 해당 파일을 바탕으로 악성코드를 탐지하는 함수를 제작하였습니다.
>>> pickleName = 'MalwareSample.p' 
>>> MalwareDetectionFunctionUsingPickle(clf, pickleName)
1
```
### BBMD/XGBClassifier_based_on_BPE_Token_word2vecEmbedding/모델 사용법
* 디렉토리 내부에 있는 test.ipynb 파일을 쥬피터 노트북을 통해 읽으시면 쉽게 이해하실 수 있습니다.
```python
>>> # 필요한 패키지인 pefile과 capstone이 설치된 환경에서 작동됩니다.
>>> # utils.py가 있는 파일로 디렉토리를 이동하시고 사용하여야 합니다.
>>> # 필요한 패키지 import
>>> from utils import *
>>> from xgboost import XGBClassifier
>>> from gensim.models import Word2Vec
>>> from gensim.models import KeyedVectors
>>> # opcodeList가 있는 파일과 BPE Token vocab 파일을 로드를 해줍니다.
>>> with open('opcodesList.txt', 'rb') as lf:
>>>     opcodes = pickle.load(lf)
>>> opcodes2 = []
>>> for i in range(len(opcodes)):
>>>    opcodes2.append([opcodes[i]])
>>> with open('2000vocab.p', 'rb') as file:
>>>    vocab = pickle.load(file)
>>> vocab = vocab + opcodes2
>>> # Trained word2vec model load
>>> modelPath = 'BPEw2v2000'
>>> word2vecModel = KeyedVectors.load_word2vec_format(modelPath)
>>> # word2vec model test
>>> model_result = word2vecModel.most_similar("addadd")
>>> print(model_result)
[('addaddadd', 0.4414026737213135), ('addaddaddadd', 0.3697534203529358), ('decaddaddadd', 0.3347220718860626), ('pushaddaddaddadd', 0.3212829530239105), ('addaddaddaddadd', 0.3178083300590515), ('adcaddadd', 0.31767505407333374), ('pushaddadd', 0.2995673418045044), ('jnpaddadd', 0.2970563769340515), ('pushaddaddadd', 0.29624754190444946), ('xchgaddadd', 0.2945404648780823)]
>>> # test.ipynb 파일 내부의 get_sentence_mean_vector(morphs) 함수를 구현하셔서 pefile을 word2vec 임베딩 값으로 변경해주는 함수를 만들어주셔야 합니다.(길어서 여기서는 생략합니다.)
>>> # 학습된 분류 모델 로드
>>> # 모델명
>>> filename = 'XGB_BPE2000_Word2vecClassifier.model'
>>> # 모델 불러오기
>>> clf = pickle.load(open(filename, 'rb'))
>>> # test.ipynb 파일 내부에 있는 MalwareDetectionFunction 함수와 MalwareDetectionFunctionUsingPickle 함수를 만들어주세요(길어서 여기서는 생략합니다.)
>>> # 양성 파일 테스트(0으로 출력될 경우 양성)
>>> fileName = '3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe'
>>> MalwareDetectionFunction(clf, fileName)
0
>>> # 악성코드 파일 테스트(1로 출력될 경우 악성)
>>> # 악성코드 파일을 직접 Google Drive 및 GitHub에 올릴 수 없기 때문에 pickle로 먼저 opcodeSequence를 추출하여 해당 파일을 바탕으로 악성코드를 탐지하는 함수를 제작하였습니다.
>>> pickleName = 'MalwareSample.p' 
>>> MalwareDetectionFunctionUsingPickle(clf, pickleName)
1
```

### BBMD/DeepLearningModel_based_on_BPETokenSequence/모델 사용법
* 디렉토리 내부에 있는 test.ipynb 파일을 쥬피터 노트북을 통해 읽으시면 쉽게 이해하실 수 있습니다.
* 해당 모델은 github에 한번에 올릴 수 있는 용량인 25MB를 초과해서 분할 압축되었습니다.(분할압축된 파일을 일괄적으로 압축해제 해주셔야 합니다.)
```python
>>> # 필요한 패키지인 pefile과 capstone이 설치된 환경에서 작동됩니다.
>>> # utils.py가 있는 파일로 디렉토리를 이동하시고 사용하여야 합니다.
>>> # 필요한 패키지 import
>>> from utils import *
>>> import keras
>>> from keras.models import Sequential
>>> from keras.layers import Dense, LSTM, Embedding
>>> from keras.layers import Conv1D, MaxPooling1D, Dropout, Activation
>>> import tensorflow.keras.backend as K
>>> from tensorflow.keras.callbacks import ModelCheckpoint
>>> from keras.preprocessing import sequence
>>> from keras.utils import np_utils
>>> from keras.utils.np_utils import to_categorical
>>> import tensorflow as tf
>>> # opcodeList가 있는 파일과 BPE Token vocab 파일을 로드를 해줍니다.
>>> with open('opcodesList.txt', 'rb') as lf:
>>>     opcodes = pickle.load(lf)
>>> print(opcodes[:10])
['aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'and', 'call', 'cbw', 'clc']
>>> opcodes2 = []
>>> for i in range(len(opcodes)):
>>>    opcodes2.append([opcodes[i]])
>>> with open('2000vocab.p', 'rb') as file:
>>>    vocab = pickle.load(file)
>>> vocab = vocab + opcodes2
>>> print(vocab[:10])
[['add', 'add'], ['mov', 'mov'], ['add', 'add', 'add', 'add'], ['push', 'push'], ['add', 'add', 'add', 'add', 'add', 'add', 'add', 'add'], ['push', 'call'], ['push', 'mov'], ['dec', 'mov'], ['mov', 'mov', 'mov', 'mov'], ['pop', 'pop']]
>>> # 모델 생성
>>> model= Sequential()
>>> model.add(Embedding(len(vocab), 3000))
>>> model.add(Dropout(0.5))
>>> model.add(Conv1D(64, 5, padding = 'valid', activation = 'relu', strides = 1))
>>> model.add(MaxPooling1D(pool_size=4))
>>> model.add(LSTM(55))
>>> model.add(Dense(48, activation='relu'))
>>> model.add(Dense(2))
>>> model.add(Activation('softmax'))
>>> model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, None, 3000)        6336000   
                                                                 
 dropout_1 (Dropout)         (None, None, 3000)        0         
                                                                 
 conv1d_1 (Conv1D)           (None, None, 64)          960064    
                                                                 
 max_pooling1d_1 (MaxPooling  (None, None, 64)         0         
 1D)                                                             
                                                                 
 lstm_1 (LSTM)               (None, 55)                26400     
                                                                 
 dense_2 (Dense)             (None, 48)                2688      
                                                                 
 dense_3 (Dense)             (None, 2)                 98        
                                                                 
 activation_1 (Activation)   (None, 2)                 0         
                                                                 
=================================================================
Total params: 7,325,250
Trainable params: 7,325,250
Non-trainable params: 0
_________________________________________________________________
>>> # 모델 가중치 로드
>>> # 모델을 저장할 경로를 설정해주시면 됩니다.
>>> filename = 'BPESequenceBasedLSTMCNNMalwareDetectionModel.h5'
>>> model.load_weights(filename)
>>> # test.ipynb 파일 내부에 있는 MalwareDetectionFunction 함수와 MalwareDetectionFunctionUsingPickle 함수를 만들어주세요(길어서 여기서는 생략합니다.)
>>> # 양성 파일 테스트(0으로 출력될 경우 양성)
>>> fileName = '3f3fe9ecad7f30fc80cdfb678d7ca27a30d0575a73818746e98be9170d3be348.exe'
>>> MalwareDetectionFunction(clf, fileName)
0
>>> # 악성코드 파일 테스트(1로 출력될 경우 악성)
>>> # 악성코드 파일을 직접 Google Drive 및 GitHub에 올릴 수 없기 때문에 pickle로 먼저 opcodeSequence를 추출하여 해당 파일을 바탕으로 악성코드를 탐지하는 함수를 제작하였습니다.
>>> pickleName = 'MalwareSample.p' 
>>> MalwareDetectionFunctionUsingPickle(clf, pickleName)
1
```



## 🌟 결과
**LSTM+CNN Model Input Max Length별 모델 성능 평가**
![KakaoTalk_20220527_170538757](https://user-images.githubusercontent.com/101659578/170662635-93601d23-33ab-45d5-b234-be2d22ff17ed.png)
* BPE Token을 2000개 생성하고 vocab을 생성하였을 경우, LSTM+CNN 모델에 돌렸을 때의 그래프입니다.
* 입력 길이를 1000개 단위로 증가시키면서 정확도를 측정한 결과, 3000이 Max length일 경우 성능이 가장 좋게 나와 해당 길이를 기준으로 정하였습니다.

**결과 테이블**
![3](https://user-images.githubusercontent.com/101659578/170676368-17e29021-25bd-4827-8260-0b2caf81af96.png)
* 기존 모델(prev.Model)과 성능을 비교한 결과 저희가 제시한 BPE 토큰을 활용한 Sequence를 입력받는 LSTM+CNN 모델이 그와 필적하는 성능을 보이는 것을 보였습니다. 이를 통해 저희는 sequence를 입력받는 BERT와 같은 타 모델을 제작할 수 있는 가능성을 확인하였고, 추후 난독화에 강하도록 pre-train이 된 모델을 직접 설계하고 제작하여 성능을 확인 할 것입니다.
* 또한 BPE Token을 기존 기법들 중 word2vec 모델과 같이 사용한 기법과 합쳐서 탐지 모델을 제작할 경우, 기존보다 더 뛰어난 성능을 보이는 것을 확인하였습니다. 이를 통해 BPE Token 생성 알고리즘을 활용하는 것이 기존 방식보다 더 좋은 성능을 내는 것을 보였습니다.


`👉 각 모델의 파일 설명은 디렉토리에 있는 read file에 기재하였습니다, 동작 과정은 각각의 ipynb의 로 설명하였습니다`

## 🔎 추가 진행 계획 및 개선 사항
* _OPCODE_ 중 _nop_ 가 아무 의미가 없는 버퍼형식으로 작동하는것을 발견하였습니다. _nop_ 를 제거하는 방식으로 성능을 높일 수 있다 판단하여, 프로젝트를 발전시켜볼 계획입니다.
* Sequence 데이터를 활용할 경우, 굉장히 긴 데이터를 모델에 Input을 더 잘반영하는 모델을 제작하기 위해 긴 데이터를 줄이는 작업이 필요하고, 악성코드 BPE 토큰들 중 불용어를 선택하기 위해 TF-IDF를 통해 값이 0인 토큰들을 불용어로 가정하는 작업들을 진행해볼 예정입니다.
* Sequence를 학습하는 모델이 뒤의 리스트를 반영하지 못한다는 단점을 해결하기 위해 추가적인 모델 활용 예정(ex) seq2seq, attention, transformer 등)입니다. 본 연구에서는 3000을 max_length로 설정하여 LSTM+CNN 모델을 제작하였을 경우에도 99%의 정확도를 확인하였지만, 뒤의 opcode를 반영하지 못하고 slice 되는 문제를 해결하기 위해 seq2seq등을 통해 뒤 리스트를 예측하도록 인코더와 디코더를 학습시킨 뒤, 인코더에서 추출한 Feature로 학습하는 모델을 제작할 예정입니다.
* TBPTT기반 순환신경망 모델을 제작 예정입니다. TBPTT는 순환신경망의 전체적인 가중치를 학습하는 과정속에서 발생되는 Gradient Descent를 방지하기 위해 사용되는 오차역전파 방식입니다. TBPTT는 자연어처리 분야에서 Performance 관점에서 뛰어난 성능을 내지 못해 잘 사용되지는 않으나, 긴 시계열 데이터를 학습하는 과정속에서 기울기 소실문제를 완전히 방지한다는 장점을 가지고 있고 본 연구에서는 이를 활용하여 성능을 측정할 예정입니다.
* 기존 모델들의 경우, 난독화가 이루어진 악성코드를 잘 예측하지 못한다는 단점이 존재합니다. 본 프로젝트는 Sequence를 입력 받는 모델의 성능이 기존 모델들과 비슷한 성능을 내는 것을 확인하였고, 이를 바탕으로 난독화된 code와 원본 code 사이의 loss를 줄이는 consistency Training과 BERT와 같이 사전 학습을 하되, 난독화와 원본 사이의 관계를 예측하게 하는 등의 Custom Pre-train 방식을 개발하여 단점을 극복할 예정입니다.
