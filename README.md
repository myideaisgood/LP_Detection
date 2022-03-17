# LP_Detection
LP_Detection

# 원본 코드

https://github.com/uvipen/Yolo-v2-pytorch


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

# TO DO

1. 다른 Dataset 도 가능하게 구축

2. 현재는 dataset resize만 함 ==> dataset crop + augmentation 추가

3. Vehicle detection 이랑 연동 ==> https://github.com/MaryamBoneh/Vehicle-Detection

4. Image size 448 고정말고 다른 image size 도 가능하게 추가 수정

등등?
