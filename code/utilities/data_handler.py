
import numpy as np

from helpers.getdata import _import_data
from helpers.preparedata import _preprocess_data, _slice_data

def import_data(path,subject=1,session=1,task=1):
    return _import_data(path,subject,session,task)            
    
def preprocess_data(raw, fparams):
    return _preprocess_data(raw, fparams)    
    
def slice_data(raw, tparams, events_interest):    
    return _slice_data(raw, tparams, events_interest)

def get_data(data_params):

    path    = data_params['path']
    subject = data_params['subject']
    session = data_params['session']
    task    = data_params['task']    
    tparams = data_params['tparams']
    fparams = data_params['fparams']

    raw, event_id = import_data(path, subject, session, task)
    raw = preprocess_data(raw, fparams)
    
    epochs = slice_data(raw, tparams, event_id)
    
    X = epochs.get_data()
    y = epochs.events[:, -1]

    return X, y        

if __name__=='__main__':
    
    data_params = {}
    data_params['path'] = '/localdata/coelhorp/datasets/motorimagery/BCI-competitions/BCI-III/IVa/'
    data_params['subject'] = 1
    data_params['session'] = 1
    data_params['task']    = 1
    data_params['tparams'] = [0.5, 2.5]
    data_params['fparams'] = [8.0, 35.0]  

    X,y = get_data(data_params)

    

           
    