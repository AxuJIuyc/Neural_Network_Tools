import pandas as pd


# +
def zero(num):
    """zero (str): number"""
    if len(num)==2:
        return num
    else:
        return f"0{num}"

    
def timeformat(s):
    """s (str): time string"""
    s = s.split(':')
    l = len(s)
    if l == 1:
        return f"00:00:{zero(s[0])}"
    elif l == 2:
        return f"00:{zero(s[0])}:{zero(s[1])}"
    elif l == 3:
        return f"{zero(s[0])}:{zero(s[1])}:{zero(s[2])}"
    else:
        return None


def table2dict(table):
    if 'EngOp' in table.columns:
        a = table.set_index(['Видео', 'EngOp'])
    else:
        a = table.set_index(['Видео', 'Операция'])
    data = {}
    for ids in a.index.unique():
        vid, cls = ids
        if vid not in data:
            data.update({f'{vid}':{}})
        if cls not in data[vid]:
            data[vid].update({f'{cls}':[]})
        a1 = a.loc[(vid, cls)][['Начало, ч:м:с', 'Конец, ч:м:с']]
        times = []
        for t1,t2 in zip(a1['Начало, ч:м:с'], a1['Конец, ч:м:с']):
            t1 = tuple(map(int, t1.split(':')))
            t2 = tuple(map(int, t2.split(':')))
            times.append([t1,t2])
        data[vid][cls] += times
    return data
    
def read_anno(filename, dct_filename=None, otype="table"):
    """
    filename (str): path to annotation file
    dct_filename (str): path to class names dictionary file
    otype (str): type of output file - 'table' or 'dict'
    """
    # Чтение файла аннотаций
    anno = pd.read_csv(filename)
    
    # Отсев пояснений
    i = []
    if anno['Видео'][0] in ["Пояснение Камера", "Пояснение Видимость"]:
        i.append(0)
    if anno['Видео'][1] in ["Пояснение Камера", "Пояснение Видимость"]:
        i.append(1)
    if len(i) != 0:
        note = anno[:len(i)] # заметки
        anno.drop(i, axis=0, inplace=True)
        anno.reset_index(drop=True, inplace=True)

    # Заполнение таблицы
    anno[anno.columns[:-1]] = anno[anno.columns[:-1]].fillna(method='ffill')
    
    # Очистка от случайных пробелов
    anno['Видео'] = anno['Видео'].apply(lambda x: x.strip())
    anno['Операция'] = anno['Операция'].apply(lambda x: x.strip().lower())
    
    # Чтение файла словаря
    if dct_filename:
        dct = pd.read_csv(dct_filename)
        dct['Перевод'] = dct['Перевод'].apply(lambda x: x.lower())
        dct.set_index('Перевод', inplace=True)
        # Перевод наименований
        anno['EngOp'] = anno['Операция'].apply(lambda x: dct.loc[x]['Имя'])
    
    # Форматирование времени 
    cols = ['Начало, ч:м:с', 'Конец, ч:м:с']
    anno[cols] = anno[cols].applymap(lambda x: timeformat(x))

    anno['Duration'] = (pd.to_datetime(anno['Конец, ч:м:с']) - 
                        pd.to_datetime(anno['Начало, ч:м:с']))
    anno['Duration'] = anno['Duration'].apply(lambda x: x.seconds)
    
    # Отсев лишних колонок
    for col in ['Примеч', 'Видимость']:
        if col in anno.columns:
            anno.drop(col, axis=1, inplace=True)
    
    # Перевод таблицы в словарь
    if otype == 'dict':
        anno = table2dict(anno)
    
    return anno


# -

if __name__ == "__main__":
    filename = "test_sum_215.csv"
    dct_filename = "test_dict.csv"
    ann = read_anno(filename, dct_filename, otype='table')


