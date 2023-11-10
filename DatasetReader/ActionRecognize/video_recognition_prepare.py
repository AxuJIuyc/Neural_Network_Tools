# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import subprocess
import ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from read_annotations import read_anno

from IPython.display import clear_output

# +
DATASET_1 = {
    'ch03_20231002080000.mp4': {'gas_analyzer':[[(0,0,36), (0,0,46)],
                                           [(0,1,36),(0,1,42)],
                                           [(0,2,21),(0,2,25)]], 
                           },
    'ch02_20231002080000.mp4': {'gas_analyzer':[[(0,3,10),(0,3,15)],
                                           [(0,3,22),(0,3,27)]]
                           },
    'ch03_20231002080315.mp4': {'syringing': [[(0,8,56),(0,11,50)]],
                                'inspection': [[(0,14,50),(0,16,38)],
                                               [(0,24,47),(0,25,37)],
                                               [(0,36,10),(0,37,10)]],
                                'measure': [[(0,25,38),(0,33,30)]],
                                'pipe_work': [[(0,41,25),(0,44,24)]],
                                'HRW_work': [[(0,45,1),(0,47,38)],
                                            [(0,49,36),(0,51,15)],
                                            [(0,54,0),(0,54,40)]],
                                'pipe_up': [[(0,49,10),(0,49,35)],
                                           [(0,53,11),(0,53,34)]],
                                'pipe_down': [[(0,51,17),(0,51,42)],
                                             [(0,53,4),(0,53,9)],
                                             [(0,54,41),(0,54,55)]]
                           }
}

# +
# format time
def ft(t): 
    if 0<=t<=9:
        t = f'0{t}'
        return t
    else: return str(t)

def format_time(h,m,s):
    h = ft(h)
    m = ft(m)
    s = ft(s)
    return h+m+s

# seconds to time
def s2t(sec):
    h = sec//3600
    dh = sec%3600
    m = dh//60
    s = dh%60
    return (h,m,s)

# time to seconds
def t2s(time):
    h,m,s = time
    return h*3600 + m*60 + s

# crop timeline to durations
def cutting(t1, t2, step):
    times = []
    while t1<=t2:
        times.append(t1)
        t1 += step
    return times

def get_seconds(times, duration):
    cuts = []
    for time in times:
        t1, t2 = time
        t1 = t2s(t1)
        t2 = t2s(t2)
        cuts.extend(cutting(t1,t2,duration))
    return cuts

# get class names from dataset
def extract_labels(dataset):
    labels_list = {}
    i = 0
    for video, labels in dataset.items():
        for label in labels.keys():
            if label not in labels_list:
                labels_list.update({label: i})
                i += 1
    return labels_list

# create txt file with class names
def create_labelstxt(labels_list, lbl_path):
    with open(lbl_path, 'w') as f:
        for label, num in labels_list.items():
            f.write(f'{label}\n')
    print(f'{lbl_path} has been created')

# create txt file with video names
def create_splittxt(video_arr, path):
    with open(path, 'w') as f:
        for name, num in video_arr:
            f.write(f'{name} {num}\n')
    print(f'{path} has been created')

# split data to train/test parts
def split_data(data, split_probability=0.8):
    '''
    data (np.array((N,2))): array of N*(video_name, num_label) of all videos
    split_probability (float): вероятность разделения (соотношение)
    '''
    # Генерируем массив случайных значений True/False с вероятностью разделения
    split_mask = np.random.rand(len(data)) < split_probability

    # Разделяем исходный массив на две части на основе маски
    train = data[split_mask]
    val = data[~split_mask]
    
    return train, val


# +
def run_video_cut_0(input_video, start_time, duration, output_file):
    command = [
        "ffmpeg",
        "-i", input_video,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c", "copy",
        "-loglevel", "quiet",
        output_file
    ]
    subprocess.run(command)
    
def run_video_cut_1(input_video, start_time, duration, output_file):
    command = [
        "ffmpeg",
        "-i", input_video,
        "-filter_complex", f"trim={start_time}:{start_time + duration}, setpts=PTS-STARTPTS",
        "-loglevel", "quiet",
        output_file
    ]
    subprocess.run(command)


# -

def video_cut(input_video, output_directory, start_times, duration=10, vctype=1):
    """
    Cutting the input video into clips of a certain duration into the output directory
    
    input_video (str): path to source video
    output_directory (str): path to output dir
    start_times (list): list of clip starting timepoints
    duration (int): clip duration
    vctype (int): type of video cutting, should be in [0, 1]

    Return video_names (list): list of videoclips names
    
    Note: 1) Use vctype=0 for fast cutting but can be troubles with short clips 
          2) Use vctype=1 for safe cutting but takes a long time
    """
    # Создаем выходную директорию, если она не существует
    subprocess.run(["mkdir", "-p", output_directory])

    # Разбиваем видео на фрагменты
    video_names = []
    for i, start_time in enumerate(start_times):
        video_name, ext = os.path.splitext(os.path.basename(input_video))
        h,m,s = s2t(start_time)
        h2,m2,s2 = s2t(start_time+duration)
        t1, t2 = format_time(h,m,s), format_time(h2,m2,s2)
        new_name = f'{video_name}_cut_{t1}_{t2}.mp4'
        output_file = f"{output_directory}/{new_name}"
        if os.path.exists(output_file):
            print(f'{output_file} is alrady exists')
            video_names.append(new_name)
            continue
        if vctype == 0:
            run_video_cut_0()
        elif vctype == 1:
            run_video_cut_1()
        else:
            raise ValueError("vctype should be in [0,1]")
        
        print(new_name, 'has been created')
        video_names.append(new_name)

    return video_names


# +
def cut_dataset(dataset, input_dir, output_dir, label_dict, duration=5, vctype=1):
    ''' 
    dataset (dict): dictionary of dataset like 
                    {'video_name_0': {'class_name_0': [[(h00,m00,s00),(h01,m01,s01)]]}}
    input_dir (str): path to folder with source videos
    output_dir (str): path to folder with cutted clips
    label_dict (dict): dictionary with class names like
                        {'class_name_0': 0, ''class_name_0': 1'}
    duration (int): Duration of each clip (seconds)
    vctype (int): type of video cutting, should be in [0, 1]
    
    Return data (np.array): array of [N*(video_name, num_label)] with all clips
    
    Note: 1) Use vctype=0 for fast cutting but can be troubles with short clips 
          2) Use vctype=1 for safe cutting but takes a long time
    '''
    
    data = np.array([[None, None]])
    for video_name, labels in tqdm(dataset.items(), desc='Processing videos', position=0):
        # Путь к исходному видео
        input_video = os.path.join(input_dir, video_name)
        
        # Список начальных времен фрагментов (в секундах)
        for label, times in tqdm(labels.items(), desc='Processing labels', position=0):
            start_times = get_seconds(times, duration)
            video_list = video_cut(input_video, output_dir, start_times, duration, vctype)
            ll = np.full(len(video_list), label_dict[label])
    
            video_list = np.column_stack((video_list,ll))
            data = np.concatenate((data, video_list))
    print("Готово! Видеофрагменты разделены и сохранены в", output_dir)
    return data[1:]

def rename_videos(folder):
    videos = os.listdir(folder)
    for video in videos:
        if os.path.splitext(video)[1] != '.mp4':
            continue
        new_name = video.replace('-', '_')
        os.rename(os.path.join(folder, video), os.path.join(folder, new_name))

if __name__ == '__main__':
    # датасет
#     dataset = dataset
    filename = "test_sum_215.csv" # marked_dataset.csv
    dct_filename = "test_dict.csv" # dictionary of class names (rus/eng)
    dataset = read_anno(filename, dct_filename, otype='dict')
    
    # Путь к исходным видео файлам
    input_dir = "../data/summary/violations/video/"
#     rename_videos(input_dir)
    
    # Путь для сохранения фрагментов
    output_dir = "../data/summary/violations/cuts1"
    
    label_dict = extract_labels(dataset)
    data = cut_dataset(dataset, input_dir, output_dir, label_dict, duration=5)
    train, val = split_data(data, 0.8)
    
    create_labelstxt(label_dict, output_dir+'_labels.txt')    
    create_splittxt(train, output_dir+'_train.txt')
    create_splittxt(val, output_dir+'_val.txt')    


# # Проверка соотношения классов

# +
def counter(dct, label):
    if label not in dct:
        dct.update({f'{label}': 1})
    else:
        dct[label] += 1
    return dct

def read_anno(path):
    with open(path, 'r') as f:
        data = f.read().split('\n')
    ratio = {}
    for row in data:
        if not row:
            continue
        video, lblnum = row.split(' ')
        ratio = counter(ratio, lblnum)
    return ratio

def check_class_ratio(train, val, labels):
    """
    train (str): path to train.txt
    val (str): path to val.txt
    labels (str): path to labels.txt
    """
    with open(labels, 'r') as f:
        ldata = f.read().split('\n')
        labels = {}
        for num, name in enumerate(ldata):
            labels.update({f'{num}': name})
    
    tdata = pd.Series(read_anno(train)).to_frame(name='train')
    vdata = pd.Series(read_anno(val)).to_frame(name='val')
    
    data = pd.concat((tdata, vdata), axis=1)
    data.fillna(0, inplace=True)
    data['class']=[labels[x] for x in data.index]
    data.reset_index(inplace=True)
    data.set_index('class', inplace=True)
    print(data)
    data.plot(kind='barh')
    
    
    
train_path = output_dir+'_train.txt'
val_path = output_dir+'_val.txt'
labels_path = output_dir+'_labels.txt'
check_class_ratio(train_path, val_path, labels_path)
# -


