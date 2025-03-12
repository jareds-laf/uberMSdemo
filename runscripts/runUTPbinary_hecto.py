from uberMS_binary.binary import runSVI
import numpy as np
import argparse
from astropy.table import Table
import os
import logging
import h5py

specNN = './models/specNN/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'
photNN = './models/photNN/'
NNtype = 'LinNet'
mistNN = './models/mistNN/mistyNN_2.3_v256_v0.h5'

def getdata():
    hecto_dir = os.path.expanduser("/data/labs/douglaslab/sofairj/data/hecto_spectra")
    hecto_filename = os.path.join(hecto_dir, "data_ngc6819_2010.0921_ngc6819_sep2010_1.7137.h5")
    f = h5py.File(hecto_filename, 'r')


    # 2080061393129929088 is the first star listed in the spectrum used here
    # 2128158910811247488 is an SB2 whose components are separated in most spectra
    # (data_ngc6811_2019.0516_hectochelle_NGC6811_2019b_1.8149.h5)
    # 2076392838230907392 is another relevant SB2 in NGC 6819
    target = str(2076392838230907392)
    spec = Table([f[target]['wave'], f[target]['flux'],
                  f[target]['eflux']],
                  names=('wave', 'flux', 'eflux'))

    # read in phot
    # create table with photometry for every filter
    phottab = f[target]['phot']

    # get the filters
    filtarr = phottab.keys()

    phot = {}
    # create a dict with {filter name: [flux magnitude, flux error]}
    for i, filter in enumerate(filtarr):
        # skip the PS_y filter because the NN is not trained on it
        if filter != 'PS_y':
            phot[filter] = [float(phottab[filter][0]),float(phottab[filter][1])]
        else:
            print('Skipping {filter} filter')

    # Need to change the names of the Gaia EDR3 filters to DR3
    # These filters are unchanged between EDR3 and DR3
    phot['GaiaDR3_BP'] = phot.pop('GaiaEDR3_BP')
    phot['GaiaDR3_RP'] = phot.pop('GaiaEDR3_RP')
    phot['GaiaDR3_G'] = phot.pop('GaiaEDR3_G')


    # spec = Table.read('data/spec.fits')
    # phot = Table.read('data/phot.dat',format='ascii')
    
    out = {}
    
    out['spec'] = {}
    out['spec']['obs_wave'] = spec['wave']
    out['spec']['obs_flux'] = spec['flux']
    out['spec']['obs_eflux'] = spec['eflux']
    
    out['phot'] = {}
    for kk in filtarr:
        phot_i = phot[filter]
        out['phot'][kk] = [phot_i[0],phot_i[1]]

    print(out['phot'])
    
    # TODO: Need parallax for 6819
    out['parallax'] = [0.67427309655216, 0.012600687]
    out['RVest'] = 7.17
    out['Avest'] = 0.09

    return out

def runTP(dospec=True,dophot=True,outputname=None,progressbar=True,version='V0',**kwargs):
    if (dospec == False) & (dophot==False):
        print('User did not set either dospec and/or dophot, returning nothing')
        return 

    # grab data
    data = getdata()

    # init input dictionary
    indict = {}
    
    if outputname is None:
        # set the output file name
        indict['outfile'] = f'./samples/samples_demo_UTPsmes_{version}.fits'
    else:
        # set the output file name
        indict['outfile'] = './samples/{}'.format(outputname)
    
    # add spec, phot, and parallax info into indict
    indict['data'] = {}
    if dospec:
        indict['data']['spec']     = data['spec']
    if dophot:
        indict['data']['phot']     = data['phot']
        indict['data']['parallax'] = data['parallax']

    # define specNN which is a list of NN files assoicated with the input data spectra
    # nspec = len(indict['data']['spec'])
    # specNN = [specNN_rv31 for _ in range(nspec)]

    distest = 1000.0/data['parallax'][0]
    distmin = 1000.0/(data['parallax'][0] + 5.0*data['parallax'][1]) 
    distmax = 1000.0/(data['parallax'][0] - 5.0*data['parallax'][1])

    print('---- Input Data ----')
    if 'phot' in indict['data'].keys():
        print('Phot:')
        for kk in indict['data']['phot'].keys():
            print('      {0} = {1:f} +/- {2:f}'.format(kk,*indict['data']['phot'][kk]))
    if 'spec' in indict['data'].keys():
        print('... For Spec:')
        spec_i = indict['data']['spec']
        print('number of pixels: {0}'.format(len(spec_i['obs_wave'])))
        print('min/max wavelengths: {0} -- {1}'.format(spec_i['obs_wave'].min(),spec_i['obs_wave'].max()))
        print('median flux: {0}'.format(np.median(spec_i['obs_flux'])))
        print('median flux error: {0}'.format(np.median(spec_i['obs_eflux'])))
        print('SNR: {0}'.format(np.median(spec_i['obs_flux']/spec_i['obs_eflux'])))
        print('      Npixels = {0}'.format(len(spec_i['obs_wave'])))
    if 'parallax' in indict['data'].keys():
        print('Parallax:')
        print('      parallax = {0} +/- {1}'.format(*indict['data']['parallax']))
        print('Distance Range:')
        print('      dist = {0} - {1} pc'.format(distmin,distmax))

    # set some initial guesses at parameters
    
    Avest    = data['Avest']
    RVest    = data['RVest']

    initpars = ({
        'Teff_a':6000.0,
        'Teff_b':5000.0,
        'log(g)_a':4.0,
        'log(g)_b':5.0,
        '[Fe/H]_a':0.0,
        '[Fe/H]_b':0.0,
        '[a/Fe]_a':0.0,
        '[a/Fe]_b':0.0,
        'log(R)_a':0.0,
        'log(R)_b':0.0,
        'mass_ratio':0.8,
        'vrad_sys':2.5,
        'vrad_a':-10.0,
        'vrad_b':10.0,
        'vstar_a':1.0,
        'vstar_b':1.0,
        'vmic_a':1.0,
        'vmic_b':1.0,
        'dist':distest,
        'Av':Avest,
        'specjitter':1E-5,
        'photjitter':1E-5
        })
    
 
    indict['initpars'] = initpars

    print('------ Init Parameters ---')
    for kk in initpars.keys():
        print('      {0} = {1}'.format(kk,initpars[kk]))

    # define priors
    indict['priors'] = {}

    # q-vrad relationship
    # indict['priors']['mass_ratio']  = ['uniform',[1e-5, 1.0]]
    indict['priors']['vrad_sys']  = ['uniform',[-500.0, 500.0]]

    for kk in ['a','b']:
        # stellar priors
        indict['priors'][f'Teff_{kk}']    = ['uniform',[2500.0,10000.0]]
        indict['priors'][f'log(g)_{kk}']  = ['uniform',[0.0,5.5]]
        # indict['priors'][f'[Fe/H]_{kk}']  = ['uniform',[-3.0,0.5]]
        # indict['priors'][f'[a/Fe]_{kk}']  = ['uniform',[-0.2,0.6]]
        indict['priors'][f'log(R)_{kk}']  = ['uniform',[-3,3]]

        # spectra priors
        # indict['priors'][f'vstar_{kk}'] = ['uniform',[0.0,250.0]]
        indict['priors'][f'vstar_{kk}'] = ['tnormal',[0.0,4.0,0.0,50.0]]
        # indict['priors'][f'vmic_{kk}']  = ['uniform',[0.5,2.0]]
        indict['priors'][f'vmic_{kk}']  = ['Bruntt2012','fixed']
        # if kk == 'a':
        #     indict['priors'][f'vrad_{kk}']  = ['uniform', [-500.0,500.0]]
        # else:
        #     indict['priors'][f'vrad_{kk}']  = ['Wilson1941', 'fixed']

    # fix chemistry to be identical
    indict['priors']['binchem'] = ['binchem','fixed']
# q-vrad priors
    indict['priors']['q_vr'] = ['Wilson1941','fixed']

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[Avest,0.1,0.0,Avest+1.0]]
    indict['priors']['dist'] = ['uniform',[distmin,distmax]] # distance in pc

    indict['priors']['lsf'] = ['tnormal',[32000.0,100.0,15000.0,40000.0]]

    indict['priors']['pc0'] = ['uniform',[0.85,1.05]]
    # indict['priors']['pc0'] = ['fixed',1.0]
    indict['priors']['pc1'] = ['uniform',[-0.25,0.25]]
    indict['priors']['pc2'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc3'] = ['uniform',[-0.05,0.05]]

    indict['priors']['specjitter'] = ['fixed',0.0]
    # indict['priors']['specjitter'] = ['tnormal',[0.0,0.001,0.0,0.01]]
    indict['priors']['photjitter'] = ['fixed',0.0]
    # indict['priors']['photjitter'] = ['tnormal',[0.0,0.001,0.0,0.01]]

    print('------ Priors -----')
    for kk in indict['priors'].keys():    
        print('       {0}: {1}'.format(kk,indict['priors'][kk]))

    # define SVI parameters
    indict['svi'] = ({
        'steps':300,
        'opt_tol':1E-6,
        'start_tol':1E-2,
        'progress_bar':progressbar,
        'post_resample':30000,
        })

    print('... Running TP')
    SVI = runSVI.sviTP(specNN=specNN,photNN=photNN,NNtype=NNtype,verbose=True)
    SVI.run(indict)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o',dest='outputname',help='output name for samples',default=None,type=str)
    parser.add_argument('--version',help='analysis run version number',type=str,default='V0')

    parser.add_argument('--progressbar',   '-pb',  dest='progressbar', action='store_true')
    parser.add_argument('--noprogressbar', '-npb', dest='progressbar', action='store_false')
    parser.set_defaults(progressbar=True)

    parser.add_argument('--dospec', '-ds', dest='dospec', action='store_true')
    parser.add_argument('--nospec', '-ns', dest='dospec', action='store_false')
    parser.set_defaults(dospec=True)

    parser.add_argument('--dophot', '-dp', dest='dophot', action='store_true')
    parser.add_argument('--nophot', '-np', dest='dophot', action='store_false')
    parser.set_defaults(dophot=True)

    args = parser.parse_args()
    runTP(**vars(args))