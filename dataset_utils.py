
# coding: utf-8

# In[ ]:



import os
import nltk
import numpy
import numpy as np
import re

import emolga.dataset.build_dataset as db


# In[1]:



sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
SENTENCEDELIMITER = '<eos>'
DIGIT = '<digit>'


# In[ ]:





def prepare_text(record, process_type=1):

    
    if process_type==0:
        
        text = record['abstract'].replace('e.g.', 'eg')
  
        title = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title'])
        sents = [re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', s) for s in sent_detector.tokenize(text)]
        text = title + ' ' + SENTENCEDELIMITER + ' ' + (' ' + SENTENCEDELIMITER + ' ').join(sents)
    elif process_type==1:
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title']) + ' '+SENTENCEDELIMITER + ' ' + re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['abstract'])
    elif process_type==2:
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['abstract'])
    return text


# In[ ]:


def get_tokens(text, process_type=1):
   
    if process_type == 0:
        text = re.sub(r'[\r\n\t]', ' ', text)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,]', text))
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    elif process_type == 1:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    elif process_type == 2:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))

    return tokens


# In[ ]:


def process_keyphrase(keyword_str):
    keyphrases = keyword_str.lower()
    keyphrases = re.sub(r'\(.*?\)', ' ', keyphrases)
    keyphrases = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', keyphrases)
    keyphrases = [filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', phrase)) for phrase in
                  keyphrases.split(';')]
    keyphrases = [[w if not re.match('^\d+$', w) else DIGIT for w in phrase] for phrase in keyphrases]

    return keyphrases


# In[ ]:



def build_data(data, idx2word, word2idx):
    Lmax = len(idx2word)
    instance = dict(source_str=[], target_str=[], source=[], target=[], target_c=[])

    for count, pair in enumerate(data):
        source, target = pair

        A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
        B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
        C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

        instance['source'] += [A]
        instance['target'] += [B]

        if count % 1000 == 0:
            print('-------------------- %d ---------------------------' % count)
            print(source)
            print(target)
            print(A)
            print(B)
            print(C)
    return instance


# In[ ]:


def load_pairs(records, process_type=1 ,do_filter=False):
    wordfreq = dict()
    filtered_records = []
    pairs = []

    import string
    printable = set(string.printable)

    for id, record in enumerate(records):
        record['keyword'] = ''.join(list(filter(lambda x: x in printable, record['keyword'])))
        record['abstract'] = ''.join(list(filter(lambda x: x in printable, record['abstract'])))
        record['title'] = ''.join(list(filter(lambda x: x in printable, record['title'])))
        text        = prepare_text(record, process_type)
        tokens      = get_tokens(text, process_type)
        keyphrases  = process_keyphrase(record['keyword'])

        for w in tokens:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1

        for keyphrase in keyphrases:
            for w in keyphrase:
                if w not in wordfreq:
                    wordfreq[w]  = 1
                else:
                    wordfreq[w] += 1

        if id % 10000 == 0 and id > 1:
            print('%d \n\t%s \n\t%s \n\t%s' % (id, text, tokens, keyphrases))
            # break

        fine_tokens = re.split(r'[\.,;]',record['keyword'].lower())
        if sum([len(k) for k in keyphrases]) != 0:
            ratio1 = float(len(record['keyword'])) / float(sum([len(k) for k in keyphrases]))
            ratio2 = float(sum([len(k) for k in fine_tokens])) / float(len(fine_tokens))
        else:
            ratio1 = 0
            ratio2 = 0
        if ( do_filter and (ratio1< 3.5)): # usually ratio1 < 3.5 is noise. actually ratio2 is more reasonable, but we didn't use out of consistency
            print('!' * 100)
            print('Error found')
            print('%d - title=%s, \n\ttext=%s, \n\tkeyphrase=%s \n\tkeyphrase after process=%s \n\tlen(keyphrase)=%d, #(tokens in keyphrase)=%d \n\tratio1=%.3f\tratio2=%.3f' % (
            id, record['title'], record['abstract'], record['keyword'], keyphrases, len(record['keyword']), sum([len(k) for k in keyphrases]), ratio1, ratio2))
            continue

        pairs.append((tokens, keyphrases))
        filtered_records.append(record)

    return filtered_records, pairs, wordfreq



# In[ ]:


def get_none_phrases(source_text, source_postag, max_len):
    np_regex = r'^(JJ|JJR|JJS|VBG|VBN)*(NN|NNS|NNP|NNPS|VBG)+$'
    np_list = []

    for i in range(len(source_text)):
        for j in range(i+1, len(source_text)+1):
            if j-i > max_len:
                continue
            if j-i == 1 and (source_text[i:j]=='<digit>' or len(source_text[i:j][0])==1):
                continue
            tagseq = ''.join(source_postag[i:j])
            if re.match(np_regex, tagseq):
                np_list.append((source_text[i:j], source_postag[i:j]))

    print('Text: \t\t %s' % str(source_text))
    print('None Phrases:[%d] \n\t\t\t%s' % (len(np_list), str('\n\t\t\t'.join([str(p[0])+'['+str(p[1])+']' for p in np_list]))))

    return np_list


# In[ ]:


if __name__ == '__main__':
    import keyphrase.config as configs
    config = configs.setup_keyphrase_all()
    test_set = db.deserialize_from_file(
        config['path'] + '/dataset/keyphrase/' + config['data_process_name'] + 'semeval.testing.pkl')
    for s_index, s_str, s_tag in zip(test_set['source'], test_set['source_str'], [[s[1] for s in d ]for d in test_set['tagged_source']]):
        get_none_phrases(s_str, s_tag, config['max_len'])

