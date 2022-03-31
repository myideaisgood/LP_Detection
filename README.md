# LP_Recognition
LP_Recognition

# 원본 코드

https://github.com/clovaai/deep-text-recognition-benchmark

# Environment 구축

```
conda env create -f lpdetection.yml
```

# Dataset Directory

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

# 주의사항
val.txt << 이게 실제 test 데이터셋
test.txt << 이건 challenging 데이터셋 

# Config
IMPORTANT!!!!!!!!!!!
중국번호판 << batch_max_length=7
한국번호판 << batch_max_length=8

imgH, imgW -> input image 를 얼마로 resize 할지
batch_max_length -> output word의 max length
pad_image -> image resize 할 때 padding 할지 말지

num_fiducial -> STN 할 때 point 몇 개 사용할지
img_color -> Gray or RGB
