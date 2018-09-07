
# coding: utf-8

# In[ ]:


import os

import numpy
import shutil

import keyphrase.config
from emolga.dataset.build_dataset import deserialize_from_file
from keyphrase.dataset.keyphrase_test_dataset import load_additional_testing_data


# In[ ]:



class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.text       = ''
        self.phrases    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s' % (self.title, self.text, str(self.phrases))


# In[ ]:




def load_text(doclist, textdir):
    for filename in os.listdir(textdir):
        with open(textdir+filename) as textfile:
            doc = Document()
            doc.name = filename[:filename.find('.txt')]

            import string
            printable = set(string.printable)

            # print((filename))
            try:
                lines = textfile.readlines()

                lines = [filter(lambda x: x in printable, l) for l in lines]

                title = lines[0].encode('ascii', 'ignore').decode('ascii', 'ignore')

                text  = (' '.join(lines[2:])).encode('ascii', 'ignore').decode('ascii', 'ignore')


                doc.title = title
                doc.text  = text
                doclist.append(doc)

            except UnicodeDecodeError:
                print('UnicodeDecodeError detected! %s' % filename )
    return doclist



# In[ ]:


def load_keyphrase(doclist, keyphrasedir):
    for doc in doclist:
        phrase_set = set()
        if os.path.exists(keyphrasedir + doc.name + '.keyphrases'):
            with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
        
        if os.path.exists(keyphrasedir + doc.name + '.keywords'):
            with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
        
        doc.phrases = list(phrase_set)
    return doclist


# In[ ]:


def get_doc(text_dir, phrase_dir):
    '''
    :return: a list of dict instead of the Document object
    '''
    doclist = []
    doclist = load_text(doclist, text_dir)
    doclist = load_keyphrase(doclist, phrase_dir)

    for d in doclist:
        print(d)

    return doclist


# In[ ]:



def export_krapivin_maui():

    config  = keyphrase.config.setup_keyphrase_all()   # load settings.

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)


    dataset = test_sets['krapivin']

    train_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/krapivin/train/'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_texts = dataset['source_str'][401:]
    train_targets = dataset['target_str'][401:]
    for i, (train_text, train_target) in enumerate(zip(train_texts,train_targets)):
        print('train '+ str(i))
        with open(train_dir+str(i)+'.txt', 'w') as f:
            f.write(' '.join(train_text))
        with open(train_dir + str(i) + '.key', 'w') as f:
            f.write('\n'.join([' '.join(t)+'\t1' for t in train_target]))

    test_dir  = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/krapivin/test/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_texts = dataset['source_str'][:400]
    test_targets = dataset['target_str'][:400]
    for i, (test_text, test_target) in enumerate(zip(test_texts,test_targets)):
        print('test '+ str(i))
        with open(test_dir+str(i)+'.txt', 'w') as f:
            f.write(' '.join(test_text))
        with open(test_dir + str(i) + '.key', 'w') as f:
            f.write('\n'.join([' '.join(t)+'\t1' for t in test_target]))




# In[ ]:



def prepare_data_cross_validation(input_dir, output_dir, folds=5):
    file_names = [ w[:w.index('.')] for w in filter(lambda x: x.endswith('.txt'),os.listdir(input_dir))]
    file_names.sort()
    file_names = numpy.asarray(file_names)

    fold_size = len(file_names)/folds

    for fold in range(folds):
        start   = fold * fold_size
        end     = start + fold_size

        if (fold == folds-1):
            end = len(file_names)

        print('Fold %d' % fold)

        test_names = file_names[start: end]
        train_names = file_names[list(filter(lambda x: x < start or x >= end, range(len(file_names))))]
        # print('test_names: %s' % str(test_names))
        # print('train_names: %s' % str(train_names))

        train_dir = output_dir + 'train_'+str(fold+1)+'/'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = output_dir + 'test_'+str(fold+1)+'/'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for test_name in test_names:
            shutil.copyfile(input_dir + test_name + '.txt', test_dir + test_name + '.txt')
            shutil.copyfile(input_dir + test_name + '.key', test_dir + test_name + '.key')
        for train_name in train_names:
            shutil.copyfile(input_dir + test_name + '.txt', train_dir + train_name + '.txt')
            shutil.copyfile(input_dir + test_name + '.key', train_dir + train_name + '.key')


