import numpy as np
import matplotlib.pyplot as plt
import json
import os


def load():
    metadata = []
    data_all = []

    for filename in os.listdir('data'):
        if filename.endswith('.json'):
            path = os.path.join('data', filename)
            with open(path) as fp:
                item = json.load(fp)
                data_all.append(item['data'])
                del item['data']
                metadata.append(list(item.values()))
    return np.array(data_all), np.array(metadata)


DIFFS = np.array([0, 0.5, 1, 2, 3])
HARMS = np.array([0, 1, 2, 3, 4])
PITCHES = np.array([220, 1000])
EXPS = np.array(['<1', '1', '1-5', '5-10', '>10'])


def get_person_jnd(data):
    return np.array([get_batch_jnd(data[data[:, 0] == pitch, 1:4]) for pitch in PITCHES])


def get_person_matrix(data):
    return np.array([get_batch_matrix(data[data[:, 0] == pitch, 1:4]) for pitch in PITCHES])


def get_dataset_jnd(data):
    return np.array([get_person_jnd(person) for person in data])


def get_batch_jnd(data):
    d = abs(data)
    jnd = [DIFFS[-int(sum(d[d[:, 1] == h][:, 2]))] for h in HARMS]
    return np.array(jnd)


def get_batch_matrix(data):
    data = abs(data)
    ordered_data = data[np.lexsort((data[:, 1], data[:, 0]))]
    values = ordered_data[:, 2]
    return np.reshape(values, [5, 5])


def plot(data):
    y_val = np.average(data, axis=0)
    y_std = np.std(data, axis=0)

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(HARMS-width/2, y_val[0], width, yerr=y_std[0])
    ax.bar(HARMS+width/2, y_val[1], width, yerr=y_std[0])
    ax.set_ylabel('JND (cents)')
    ax.set_xlabel('Harmonics')
    ax.legend(['$f=220$', '$f=1000$'])
    ax.set_title('JND vs tone harmonics')

    plt.show()


def analytics_sex(meta):
    res = np.unique(meta[:, 0], return_counts=True)
    plt.pie(res[1], labels=res[0], autopct='%1.1f%%')


def analytics_exp(meta):
    res = np.unique(meta[:, 2], return_counts=True)
    x = np.array([1, 2, 3, 0, 4])

    fig, ax = plt.subplots()
    ax.bar(x, res[1])
    plt.xticks(x, res[0])
    ax.set_ylabel('Number of people')
    ax.set_xlabel('Musical experience (years)')
    ax.set_title('Distribution over musical experience')

    plt.show()
