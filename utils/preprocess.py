import os
import json

province = ['대구서', '동대문', '미추홀', '서대문', '영등포', '인천서', '인천중',
                    '강남', '강서', '강원', '경기', '경남', '경북', '계양', '고양', '관악', '광명', '광주', '구로', '금천', '김포', '남동', 
                    '대구', '대전', '동작', '부천', '부평', '서울', '서초', '안산', '안양', '양천', '연수', '용산', '인천', '전남', '전북', 
                    '충남', '충북', '영']

province_replace = ['괅', '놝', '돩', '랅', '맑', '밝', '삵', '앍', '잙', '찱',
                    '괉', '놡', '돭', '랉', '맕', '밡', '삹', '앑', '잝', '찵',
                    '괋', '놣', '돯', '뢇', '맗', '밣', '삻', '앓', '잟', '찷',
                    '괇', '놟', '돫', '뢃', '맓', '밟', '삷', '앏', '잛', '찳']

def encode_province(text):

    for idx in range(len(province)):
        prov = province[idx]
        if prov in text:
            text = text.replace(prov, province_replace[idx])

    return text

def decode_province(text):

    for idx in range(len(province)):
        prov = province_replace[idx]
        if prov in text:
            text = text.replace(prov, province[idx])
    return text

def isKorean(s):
    return not s.isascii()

def get_chars(DATA_DIR, subset):

    provinces = []

    IMG_DIR = 'image'
    LABEL_DIR = 'label'

    # Read Label path
    label_path = []

    for file in os.listdir(os.path.join(DATA_DIR, subset, LABEL_DIR)):
        if file.endswith('.json'):
            label_path.append(file)

    label_path = sorted(label_path)

    # Read all Image path & Label
    img_paths = []
    labels = []

    for path in label_path:
        cur_path = os.path.join(DATA_DIR, subset, LABEL_DIR, path)
        f = json.load(open(cur_path))

        img_paths.append(f['imagePath'])
        labels.append(f['value'])


    img_paths = []
    labels = []
    total_num = 0

    # Obtain Image & Label without provinces
    for l_path in label_path:
        cur_path = os.path.join(DATA_DIR, subset, LABEL_DIR, l_path)
        f = json.load(open(cur_path))


        if not '-' in f['value']:
            if not '미주홀' in f['value']:
                if ' ' in f['value']:
                    fixed = f['value'].replace(' ','')
                    fixed = encode_province(fixed)
                    img_paths.append(f['imagePath'])
                    labels.append(fixed)
                else:
                    fixed = encode_province(f['value'])
                    img_paths.append(f['imagePath'])
                    labels.append(fixed)

        
        total_num += 1

    # Obtain character list 'without' provinces
    chars = []
    max_length = 0

    for idx, label in enumerate(labels):
        cur_chars = list(label)
        
        for char in cur_chars:
            if not char in chars:
                chars.append(char)

        if len(cur_chars) > max_length:
            max_length = len(cur_chars)

    chars = sorted(chars)

    return chars, max_length


t_chars, t_max_length = get_chars('../DATASET/KorLP', 'Training')
v_chars, v_max_length = get_chars('../DATASET/KorLP', 'Validation')

print(sorted(t_chars))
print('Training Char num : ', len(t_chars))
print('Max Length : ', t_max_length)

print(sorted(v_chars))
print('Validation Char num : ', len(v_chars))
print('Max Length : ', v_max_length)

total_chars = sorted(list(set(t_chars) | set(v_chars)))
total_max_length = max(t_max_length, v_max_length)

print(total_chars)
print('Total Char num : ', len(total_chars))
print('Total Max Length : ', total_max_length)

char_w_province = sorted([x for x in total_chars if x not in province_replace])
print(char_w_province)
print(len(char_w_province))