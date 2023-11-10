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
    
    
if __name__ == "__main__":    
  train_path = output_dir+'_train.txt'
  val_path = output_dir+'_val.txt'
  labels_path = output_dir+'_labels.txt'
  check_class_ratio(train_path, val_path, labels_path)
