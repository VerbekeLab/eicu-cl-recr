import os
import json

from data_extraction.timeseries import timeseries_main
from data_extraction.flat_and_labels import flat_and_labels_main
from data_extraction.split_train_test import split_train_test

# with open('paths.json', 'r') as f:
#     eICU_path = json.load(f)["/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_Data"]

eICU_path = "/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_Data/"

if __name__=='__main__':
    print('==> Removing the stays.txt file if it exists...')
    try:
        os.remove(eICU_path + 'stays.txt')
    except FileNotFoundError:
        pass
    timeseries_main(eICU_path, test=False)
    flat_and_labels_main(eICU_path)
    split_train_test(eICU_path, is_test=False)