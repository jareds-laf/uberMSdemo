from uberMS import runSVI
import numpy as np
import argparse
from astropy.table import Table
from Payne.jax.fitutils import airtovacuum
import os
import astropy.units as u
from astropy.units import cds
from astropy.nddata import StdDevUncertainty
import h5py
from specutils import Spectrum1D

specNN = './models/specNN/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'
photNN = './models/photNN/'
NNtype = 'LinNet'
mistNN = './models/mistNN/mistyNN_2.3_v256_v0.h5'

def getdata():
    # define the output dict to start placing things in it
    out = {}
    
    # read in spectrum
    # spec = Table.read('data/demospec_18Sco.fits',format='fits')
    # spec = spec[(spec['wave'] > 5150.0) & (spec['wave'] < 5300.0)]
    
    # spec['wave'] = airtovacuum(spec['wave'])




    # TODO: Make this work with the rest of the code!
    hecto_dir = os.path.expanduser("/data/labs/douglaslab/sofairj/data/hecto_spectra")
    hecto_filename = os.path.join(hecto_dir,"data_ngc6811_2019.0516_hectochelle_NGC6811_2019b_1.8149.h5")
    f = h5py.File(hecto_filename, 'r')
    # target is a GAIA ID, likely still as an int

    # Running with GAIA ID 2080061393129929088
    target = str(2080061393129929088)
    wav = f[target]["wave"] * u.AA
    flu = f[target]["flux"]*u.Jy
    err = StdDevUncertainty(f[target]["eflux"]*u.Jy)
    spec = Spectrum1D(spectral_axis=wav, flux=flu, uncertainty=err)
    




    # normalize the spectrum (and error) by the median
    medflux = np.nanmedian(spec['flux'])
    spec['flux']  = spec['flux'] / medflux
    spec['err'] = spec['err'] / medflux
    
    # read in phot
    phottab = Table.read('data/demophot_18Sco.fits',format='fits')
    # filtarr = list(phottab['band'])
    filtarr = ['GaiaDR3_G','GaiaDR3_BP','GaiaDR3_RP','2MASS_J','2MASS_H','2MASS_Ks','WISE_W1','WISE_W2']
    phot = {}
    for ii,pb in enumerate(filtarr):
        phot[pb] = [float(phottab['mag'][ii]),float(phottab['err'][ii])]
    
    # store a priori stellar parameters [from GBS website]
    out['Teff']   = 5810.0
    out['log(g)'] = 4.44
    out['[Fe/H]'] = 0.1
    out['[a/Fe]'] = 0.0
    out['log(R)'] = 0.0
    
    # set parallax, taken from SIMBAD
    out['parallax'] = [70.7371,0.0631]
    
    # put in spectrum
    out['spec'] = {}
    wave  = spec['wave']
    flux  = spec['flux']
    eflux = spec['err']
        
    # set the arrays into the output dictionary
    out['spec']['obs_wave']  = wave
    out['spec']['obs_flux']  = flux
    out['spec']['obs_eflux'] = eflux
    
    # put in photometry
    out['phot'] = phot
    out['phot_filtarr'] = filtarr

    # add vhelio just so that things work that expect a heliocentric correction
    out['Vhelio'] = 0.0

    return out

def runTPdemo(dospec=True,dophot=True,outputname=None,progressbar=True,**kwargs):
    if (dospec == False) & (dophot==False):
        print('User did not set either dospec and/or dophot, returning nothing')
        return 

    # grab data
    data = getdata()

    # init input dictionary
    indict = {}
    
    if outputname is None:
        # set the output file name
        indict['outfile'] = './samples/demoUTP.fits'
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

    print('---- Input Data ----')
    if 'phot' in indict['data'].keys():
        print('Phot:')
        for kk in indict['data']['phot'].keys():
            print('      {0} = {1:f} +/- {2:f}'.format(kk,*indict['data']['phot'][kk]))
    if 'spec' in indict['data'].keys():
        print('Spectrum:')
        print('      SNR = {}'.format(np.median(data['spec']['obs_flux']/indict['data']['spec']['obs_eflux'])))
        print('      Wavelength Range = {0}-{1}'.format(indict['data']['spec']['obs_wave'].min(),indict['data']['spec']['obs_wave'].max()))
        print('      Npixels = {0}'.format(len(indict['data']['spec']['obs_wave'])))
    if 'parallax' in indict['data'].keys():
        print('Parallax:')
        print('      parallax = {0} +/- {1}'.format(*indict['data']['parallax']))

    # set some initial guesses at parameters
    
    teffest = data['Teff']
    fehest  = data['[Fe/H]']
    afeest  = data['[a/Fe]']
    loggest  = data['log(g)']
    logRest  = data['log(R)']
    
    initpars = ({
        'Teff':teffest,
        '[Fe/H]':fehest,
        '[a/Fe]':afeest,
        'log(g)':loggest,
        'log(R)':logRest,
        'dist':1000.0/data['parallax'][0],
        'Av':0.01,
        'vmic':1.0,
        'vstar':2.0,
        'vrad':0.0,
        'pc0':1.0,
        'pc1':0.0,
        'pc2':0.0,
        'pc3':0.0,        
        'lsf':30000.0,
        'photjitter':1E-5,
        'specjitter':1E-5,            
        })
    indict['initpars'] = initpars

    print('------ Init Parameters ---')
    for kk in initpars.keys():
        print('      {0} = {1}'.format(kk,initpars[kk]))
        

    # define priors
    indict['priors'] = {}

    # stellar priors
    indict['priors']['Teff']    = ['uniform',[2500.0,10000.0]]
    indict['priors']['log(g)']  = ['uniform',[0.0,5.5]]
    indict['priors']['[Fe/H]']  = ['uniform',[-3.0,0.5]]
    indict['priors']['[a/Fe]']  = ['uniform',[-0.2,0.6]]
    indict['priors']['log(R)']  = ['uniform',[-3,3]]

    # spectra priors
    indict['priors']['vrad'] = ['uniform',[-20,20]]
    indict['priors']['vstar'] = ['uniform',[0.0,250.0]]
    indict['priors']['vmic']  = ['uniform',[0.5,2.0]]

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[0.0,0.01,0.0,0.5]]
    indict['priors']['dist'] = ['uniform',[1.0,50.0]] # distance in pc
    
    # calibration priors
    indict['priors']['lsf'] = ['tnormal',[32000.0,500.0,15000.0,40000.0]]

    indict['priors']['pc0'] = ['uniform',[0.75,3.0]]    
    # indict['priors']['pc0'] = ['fixed',1.0]
    indict['priors']['pc1'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc2'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc3'] = ['uniform',[-0.1,0.1]]

    # indict['priors']['specjitter'] = ['uniform',[1E-6,1E-2]]
    # indict['priors']['photjitter'] = ['uniform',[1E-6,1E-2]]
    indict['priors']['specjitter'] = ['fixed',0.0]
    indict['priors']['photjitter'] = ['fixed',0.0]

    print('------ Priors -----')
    for kk in indict['priors'].keys():    
        print('       {0}: {1}'.format(kk,indict['priors'][kk]))

    # define SVI parameters
    indict['svi'] = ({
        'steps':30000,
        'opt_tol':1E-6,
        'start_tol':1E-2,
        'progress_bar':progressbar,
        'post_resample':30000,
        })

    print('... Running TP')
    SVI = runSVI.sviTP(specNN=specNN,photNN=photNN,verbose=True)
    SVI.run(indict)
    
def runMSdemo(dospec=True,dophot=True,outputname=None,progressbar=True,**kwargs):
    if (dospec == False) & (dophot==False):
        print('User did not set either dospec and/or dophot, returning nothing')
        return 

    # grab data
    data = getdata()

    # init input dictionary
    indict = {}
    
    if outputname is None:
        # set the output file name
        indict['outfile'] = './samples/demoUMS.fits'
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

    print('---- Input Data ----')
    if 'phot' in indict['data'].keys():
        print('Phot:')
        for kk in indict['data']['phot'].keys():
            print('      {0} = {1:f} +/- {2:f}'.format(kk,*indict['data']['phot'][kk]))
    if 'spec' in indict['data'].keys():
        print('Spectrum:')
        print('      SNR = {}'.format(np.median(data['spec']['obs_flux']/indict['data']['spec']['obs_eflux'])))
        print('      Wavelength Range = {0}-{1}'.format(indict['data']['spec']['obs_wave'].min(),indict['data']['spec']['obs_wave'].max()))
        print('      Npixels = {0}'.format(len(indict['data']['spec']['obs_wave'])))
    if 'parallax' in indict['data'].keys():
        print('Parallax:')
        print('      parallax = {0} +/- {1}'.format(*indict['data']['parallax']))

    # set some initial guesses at parameters
    initpars = ({
        'EEP':350.0,
        'initial_Mass':1.0,
        'initial_[Fe/H]':0.0,
        'initial_[a/Fe]':0.0,
        'dist':1000.0/data['parallax'][0],
        'Av':0.01,
        'vmic':1.0,
        'vstar':2.0,
        'vrad':0.0,
        'pc0':1.0,
        'pc1':0.0,
        'pc2':0.0,
        'lsf':20000.0,
        'photjitter':1E-5,
        'specjitter':1E-5,            
        })
    indict['initpars'] = initpars

    print('------ Init Parameters ---')
    for kk in initpars.keys():
        print('      {0} = {1}'.format(kk,initpars[kk]))
    
    # define priors
    indict['priors'] = {}

    # isochrone priors
    indict['priors']['EEP'] = ['uniform',[200,500]]
    indict['priors']['initial_Mass']   = ['IMF',{'mass_le':0.75,'mass_ue':1.25}]
    # indict['priors']['initial_Mass'] = ['uniform',[0.25,2.25]]
    # indict['priors']['initial_[Fe/H]'] = ['uniform',[-3.0,0.4]]
    # indict['priors']['initial_[a/Fe]'] = ['uniform',[-0.15,0.55]]
    indict['priors']['initial_[Fe/H]'] = ['tnormal',[0.0,0.01,-3.0,0.4]]
    indict['priors']['initial_[a/Fe]'] = ['tnormal',[0.0,0.01,-0.15,0.55]]

    # spectra priors
    indict['priors']['vrad'] = ['uniform',[-20,20]]
    indict['priors']['vstar'] = ['uniform',[0.0,250.0]]
    indict['priors']['vmic']  = ['uniform',[0.5,2.0]]

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[0.0,0.01,0.0,0.5]]
    indict['priors']['dist'] = ['uniform',[1.0,50.0]] # distance in pc
    
    # calibration priors
    indict['priors']['lsf'] = ['tnormal',[32000.0,500.0,15000.0,40000.0]]

    indict['priors']['pc0'] = ['uniform',[0.75,3.0]]    
    # indict['priors']['pc0'] = ['fixed',1.0]
    indict['priors']['pc1'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc2'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc3'] = ['uniform',[-0.1,0.1]]

    # indict['priors']['specjitter'] = ['uniform',[1E-6,1E-2]]
    # indict['priors']['photjitter'] = ['uniform',[1E-6,1E-2]]
    indict['priors']['specjitter'] = ['fixed',0.0]
    indict['priors']['photjitter'] = ['fixed',0.0]

    print('------ Priors -----')
    for kk in indict['priors'].keys():    
        print('       {0}: {1}'.format(kk,indict['priors'][kk]))

    # define SVI parameters
    indict['svi'] = ({
        'steps':30000,
        'opt_tol':1E-6,
        'start_tol':1E-2,
        'progress_bar':progressbar,
        'post_resample':30000,
        })

    print('... Running MS')
    SVI = runSVI.sviMS(specNN=specNN,photNN=photNN,mistNN=mistNN,verbose=True,usegrad=False)
    SVI.run(indict)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o',dest='outputname',help='output name for samples',default=None,type=str)

    parser.add_argument('--runtype','-t',dest='runtype',help='Type of run [UTP/UMS]',default='UTP',type=str,choices=['UTP','UMS'])

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

    if args.runtype == 'UTP':
        runTPdemo(**vars(args))
    if args.runtype == 'UMS':
        runMSdemo(**vars(args))

