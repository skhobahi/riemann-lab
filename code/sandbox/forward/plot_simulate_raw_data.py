import numpy as np
import math
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw, simulate_stc

def gen_ar2(r, f, N):
    N_burn = 1000    
    x  = np.zeros(N+N_burn)
    u  = np.random.randn(N+N_burn)
    for n in range(N+N_burn):
        x[n] = 2*r*np.cos(2*np.pi*f)*x[n-1] - r**2*x[n-2] + u[n]
    x = x[N_burn:]
    x = x - np.mean(x)        
    x = x/np.sqrt(np.var(x))
    return 25e-9*x

def sigmoid(x, a, b):
    return 1.0/(math.exp(-a*(x-b)) + 1)
    
def gen_mvar(coeffs, Ns, sig=1.0, Nb=1000):
    
    if type(coeffs[0]) is list:
        m,m = coeffs[0][0].shape
        P = len(coeffs[0])     
        coeffs = [np.copy(coeffs[0]) for _ in range(Nb)] + coeffs 
    else:
        m,m = coeffs[0].shape
        P  = len(coeffs)
        coeffs = [np.copy(coeffs) for _ in range(Ns+Nb)] 
               
    x  = np.zeros((m,Ns+Nb))
    u  = np.random.randn(m,Ns+Nb)    
    for n in range(Ns+Nb):
        for p in range(P):
            x[:,n] += np.dot(coeffs[n][p], x[:,n-(p+1)]) 
        x[:,n] += sig*u[:,n]
    x = x[:,Nb:] 
    
    for i in range(m):
        x[i,:] = x[i,:] - np.mean(x[i,:])
        x[i,:] = x[i,:]/np.sqrt(np.var(x[i,:]))
    
    return 25e-9*x       

def signal_generator(n_sources, times):     
        
    r = 0.95
    f = 0.2
    Ns = 4096
    
    signal = np.zeros((n_sources, Ns))
    signal[0,:] = 0*gen_ar2(r, f, Ns)     
    signal[1,:] = gen_ar2(r, f, Ns)     
    signal[2,:] = 0*gen_ar2(r, f, Ns)     
    signal[3,:] = gen_ar2(r, f, Ns)         
        
    return signal    
    
def simulate_eeg(label_names, signal_generator, 
                 epoch_duration=3.0, n_trials=10):
    
    # Getting the paths to filenames of dataset
    data_path   = sample.data_path()
    raw_fname   = data_path + '/MEG/sample/sample_audvis_raw.fif'
    trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
    bem_fname   = (data_path +
                   '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')    
    
    # Load real data as the template
    raw = mne.io.read_raw_fif(raw_fname)
    raw = raw.crop(0., n_trials*epoch_duration)
    
    # Loading parameters for the forward solution
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, ref_meg=False,
                                 exclude=raw.info['bads']) 
        
    # Get the labels and their centers                  
    labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
              for ln in label_names] 
    subjects_dir = data_path + '/subjects'
    hemi_to_ind = {'lh': 0, 'rh': 1}          
    for i, label in enumerate(labels):
        # The `center_of_mass` function needs labels to have values.
        labels[i].values.fill(1.)
    
        # Restrict the eligible vertices to be those on the surface under
        # consideration and within the label.
        surf_vertices = fwd['src'][hemi_to_ind[label.hemi]]['vertno']
        restrict_verts = np.intersect1d(surf_vertices, label.vertices)
        com = labels[i].center_of_mass(subject='sample',
                                       subjects_dir=subjects_dir,
                                       restrict_vertices=restrict_verts,
                                       surf='white')
    
        # Convert the center of vertex index from surface vertex list 
        # to Label's vertex list.
        cent_idx = np.where(label.vertices == com)[0][0]
    
        # Create a mask with 1 at center vertex and zeros elsewhere.
        labels[i].values.fill(0.)
        labels[i].values[cent_idx] = 1.
    
    n_labels = len(label_names)    
    times  = raw.times[:int(raw.info['sfreq'] * epoch_duration)]       
    signal = signal_generator(n_labels, times)      
    
    # Generate the sources in each label
    dt  = times[1] - times[0]
    stc = simulate_stc(fwd['src'], labels, signal, times[0], dt,
                       value_fun=lambda x: x)
    
    # Simulate raw data
    raw_sim = simulate_raw(raw, stc, trans_fname, fwd['src'], bem_fname, 
                           cov='simple', iir_filter=[2, -2, 0.4], 
                           ecg=False, blink=False, n_jobs=1, verbose=True)
    
    # Get just the EEG data and the stimulus channel
    raw_eeg = raw_sim.load_data().pick_types(meg=False, eeg=True, stim=True)
    
    return raw_eeg

#%%#########################################################################

label_names = ['Aud-lh', 'Aud-rh', 'Vis-lh', 'Vis-rh']
epoch_duration = 3.0
n_trials = 20
raw_eeg  = simulate_eeg(label_names=label_names, 
                       signal_generator=signal_generator, 
                       epoch_duration=epoch_duration, 
                       n_trials=n_trials)

events = find_events(raw_eeg) 
epochs = Epochs(raw_eeg, events, event_id=1, 
                tmin=0.00, tmax=epoch_duration, baseline=None)
epochs.load_data()
evk = epochs.average()
evk.plot()

sfreq_downsample  = 150
epochs_downsample = epochs.resample(sfreq_downsample, npad='auto')
epochs_downsample = epochs_downsample.pick_types(eeg=True)

#%%

from pyriemann.estimation import Covariances

x = epochs_downsample.get_data()
Nt,Nc,Ns = x.shape
covs = Covariances().fit_transform(x)
C = np.mean(covs, axis=0)

plt.imshow(C)

#%%


def gen_windows(L, Ns, step=1):
    return np.stack([w + np.arange(L) for w in range(0,Ns-L+1, step)])
    
L  = 16
st = 1
Cw = []
for w in gen_windows(L, Ns, step=st):
    xw = x[:,:,w]
    covs = Covariances().fit_transform(xw)
    Cw.append(np.mean(covs, axis=0))   
    
#%%    

from pyriemann.utils.distance import distance_riemann
stat  = Cw[:50]
Cstat = np.mean(np.stack(stat), axis=0)
dist = []    
for Cwi in Cw:
    dist.append(np.linalg.norm(Cstat - Cwi))
plt.plot(L/2+np.arange(0, Ns-L+1),dist) 














