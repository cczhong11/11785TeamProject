#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
from collections import defaultdict, OrderedDict

GOOGLE_PRETRAINED_PATH = '/Users/zhangdu/OneDrive/Study/Graduate_Study/CMU/Class/11785/GP/GoogleNews-vectors-negative300.bin'

if __name__ == '__main__':
    vid_words = []
    with open("../resources/map_vid.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid_words.append(line.split()[-1])
    coco_names = []
    with open("../data/coco.names", "r") as f:
        lines = f.readlines()
        for line in lines:
            coco_names.append(line.split()[0])
    vid_names_dict = defaultdict(lambda: [])
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_PRETRAINED_PATH, binary=True)

    for vid_word in vid_words:
        for coco_name in coco_names:
            try:
                simi = model.similarity(vid_word, coco_name)
                vid_names_dict[vid_word].append([coco_name, simi])
            except:
                print(vid_word, coco_name)
        vid_names_dict[vid_word].sort(key=lambda x: x[1], reverse=True)
    vid_names_dict = OrderedDict(vid_names_dict)
    with open("sim_result.txt", "w") as f:
        for word, sims in vid_names_dict.items():
            out_str = [word]
            for sim in sims:
                out_str.append(sim[0])
                out_str.append(str(sim[1]))
            out_str = ' '.join(out_str) + '\n'
            f.write(out_str)
