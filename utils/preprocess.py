import os
import json

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

    # Obtain provinces
    provinces = []

    for label in labels:
        cur_chars = list(label)
        cur_length = len(cur_chars)

        cur_province = ""
        for idx in range(cur_length):
            if isKorean(cur_chars[idx]):
                cur_province += cur_chars[idx]
            else:
                break
        
        if not cur_province in provinces:
            provinces.append(cur_province)

    provinces.remove("")

    img_paths = []
    labels = []
    total_num = 0
    province_num = 0
    # Obtain Image & Label without provinces
    for l_path in label_path:
        cur_path = os.path.join(DATA_DIR, subset, LABEL_DIR, l_path)
        f = json.load(open(cur_path))

        isProvince = bool([ele for ele in provinces if(ele in f['value'])])

        if not isProvince:
            if not '-' in f['value']:
                if ' ' in f['value']:
                    fixed = f['value'].replace(' ','')
                    img_paths.append(f['imagePath'])
                    labels.append(fixed)
                else:
                    img_paths.append(f['imagePath'])
                    labels.append(f['value'])
            province_num += 1
        
        total_num += 1
    
    province_num = total_num - province_num
    print('%.1f  [%d/%d]' % (province_num/total_num*100, province_num, total_num))


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

    return provinces, chars, max_length


t_provinces, t_chars, t_max_length = get_chars('../DATASET/KorLP', 'Training')
v_provinces, v_chars, v_max_length = get_chars('../DATASET/KorLP', 'Validation')

print(t_provinces)
print(t_chars)
print('Training Char num : ', len(t_chars))
print('Max Length : ', t_max_length)

print(v_provinces)
print(v_chars)
print('Validation Char num : ', len(v_chars))
print('Max Length : ', v_max_length)

total_provinces = sorted(list(set(t_provinces) | set(v_provinces)))
total_chars = sorted(list(set(t_chars) | set(v_chars)))
total_max_length = max(t_max_length, v_max_length)

print(total_provinces)
print(total_chars)
print('Total Char num : ', len(total_chars))
print('Total Max Length : ', total_max_length)