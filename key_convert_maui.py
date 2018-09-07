
# coding: utf-8

# In[ ]:




import os
import re



if __name__ == '__main__':
    UNCONTR_DIR = "/home/memray/Project/keyphrase/maui/src/test/resources/data/Inspec/uncontr/"
    OUTPUT_DIR = "/home/memray/Project/keyphrase/maui/src/test/resources/data/Inspec/key/"


    for file_name in os.listdir(UNCONTR_DIR):
        file_no = file_name[:file_name.find('.')]
        with open(UNCONTR_DIR+file_name) as file:
            str = ' '.join(file.readlines()).replace('\n',' ').replace('\t',' ')
        keywords = [key.strip() for key in str.split(';')]

        with open(OUTPUT_DIR+file_no+'.key', 'w') as output_file:
            for keyword in keywords:
                keyword = re.sub('\s+',' ',keyword)
                print(keyword+'\t1\n')
                output_file.write(keyword+'\t1\n')

