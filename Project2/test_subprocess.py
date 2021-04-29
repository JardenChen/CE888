import os
from multiprocessing import Pool
from multiprocessing import Process

from transformers import AutoTokenizer, AutoModel, TFAutoModel
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL = "twitter-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model = model.to(device)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_embedding(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    features = model(**encoded_input)
    features = features[0].detach().cpu().numpy() 
    features_mean = np.mean(features[0], axis=0) 
    return features_mean

def loop(filename):
    datadir = './datasets/' + filename
    datapaths = os.listdir(datadir)
    for path in datapaths:
        if 'text' in path:
            datapath = os.path.join(datadir, path).replace('\\', '/')
            embeddingpath = os.path.join(datadir, path[0:path.index('text')] + 'embedding.csv').replace('\\', '/')
            # with open(embeddingpath, 'w') as out:
            #     out.write('hello')
            text = open(datapath, 'r', encoding='utf8').read().split('\n')
            embeds = np.array([get_embedding(t) for t in text])
            np.savetxt(embeddingpath, embeds, delimiter=',')
            print(filename)
        
if __name__ == '__main__':
    record = []
    for name in ['emoji', 'hate', 'irony']:
        t = Process(target=loop, args=(name, ))
        t.start()
        record.append(t)

    for t in record:
        t.join()