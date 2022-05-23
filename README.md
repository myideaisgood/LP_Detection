# LP_Recognition
LP_Recognition

# 원본 코드

https://github.com/clovaai/deep-text-recognition-benchmark

# Environment 구축

```
conda env create -f lpdetection.yml
```

# Dataset Directory

중국번호판 (CCPD)
```
|── DATASET
    ├──> CCPD2019
         ├──> ccbd_base
         ├──> ccpd_blur
         ...
         ├──> ccpd_weather
         └──> splits
                ├──> ccpd_blur.txt
                ├──> ccpd_challenge.txt
                ...
                ├──> train.txt
                ├──> val.txt
                └──> test.txt
```

한국번호판 (AI 허브 공개데이터셋)
```
|── DATASET
    ├──> KorLP
         ├──> Training
                ├──> image
                └──> label
         └──> Validation
                ├──> image
                └──> label
```

한국번호판 (제주도 자율주행)
```
|── DATASET
    ├──> jeju
         ├──> 0
         ├──> 1
         ...
         └──> 9
                ├──> origin
                └──> txt
```

# 각 Python 파일 설명
config.py -> configuration 조절  
evaluate_jeju.py -> 제주도 데이터셋에 대해 evaluate (evaluate 보다는 inference)  
evaluate.py -> CCPD / KorLP 에 evaluate  
inference.py -> GT 없는 데이터셋에 대해 inference  
kor_char_information.py -> preprocess 한 한글 글자 결과  
model.py -> 모델  
save_failure.py -> Evaluation 결과 틀린 이미지들 저장  
train.py -> CCPD/ KorLP 학습  

```
학습진행 : train_ccpd.py    or   train_kor.py
Evaluation 진행 : evaluate.py
```

# 주의사항
CCPD 에서
val.txt << 이게 실제 test 데이터셋  
test.txt << 이건 challenging 데이터셋  

# Config
IMPORTANT!!!!!!!!!!!  
중국번호판 << batch_max_length=7  
한국번호판 << batch_max_length=9

imgH, imgW -> input image 를 얼마로 resize 할지 (Default 64/200)   
batch_max_length -> output word의 max length  (!!!! Important !!! 중국번호판=7, 한국번호판=9)   
pad_image -> image resize 할 때 padding 할지 말지 (Default False)   
   
Transformation / FeatureExtraction / SequenceModeling / Prediction -> Default : TPS / ResNet / BiLSTM / CTC   

num_fiducial -> STN 할 때 point 몇 개 사용할지  (Default 8)   
img_color -> Gray or RGB   
