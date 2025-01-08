from uberMS.dva import runSVI
import numpy as np
import argparse
from astropy.table import Table
from Payne.jax.fitutils import airtovacuum
import os
import h5py
import glob

specNN_rv31 = './models/specNN/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'
photNN = './models/photNN/'
NNtype = 'LinNet'
mistNN = './models/mistNN/mistyNN_2.3_v256_v0.h5'


def getdata():
    # define the output dict to start placing things in it
    out = {}
    

    # read in spectrum
    hecto_dir = os.path.expanduser("/data/labs/douglaslab/sofairj/data/hecto_spectra")
    
    # 2080061393129929088 is the first star listed in the spectra used here
    hecto_filenames = glob.glob(os.path.join(hecto_dir, "*.h5"))
    target = str(2080061393129929088)


    specs = []
    for i, file in enumerate(hecto_filenames):
        f = h5py.File(file, 'r')

        spec = Table([f[target]['wave'], f[target]['flux'], \
                    f[target]['eflux']], \
                    names=('wave', 'flux', 'eflux'))
        spec = spec[(spec['wave'] > 5150.0) & (spec['wave'] < 5300.0)]

        # change wavelength to vac
        medflux = np.nanmedian(spec['flux'])
        spec['flux'] = spec['flux'] / medflux
        spec['eflux'] = spec['eflux'] / medflux

        specs.append(spec)

    spec1 = specs[0]
    spec2 = specs[1]
    spec3 = specs[2]
    del(specs)

    
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

    # store a priori stellar parameters [from GBS website]
    out['Teff']   = 5414.0
    out['log(g)'] = 4.49
    out['[Fe/H]'] = -0.5
    out['[a/Fe]'] = 0.3
    out['log(R)'] = 0.0
    
    # set parallax, taken from SIMBAD
    out['parallax'] = [273.8097,0.1701]
    
    # put in spectrum
    out1 = {}
    out1['obs_wave']  = spec1['wave']
    out1['obs_flux']  = spec1['flux']
    out1['obs_eflux'] = spec1['eflux']
        
    out2 = {}
    out2['obs_wave']  = spec2['wave']
    out2['obs_flux']  = spec2['flux']
    out2['obs_eflux'] = spec2['eflux']

    out3 = {}
    out3['obs_wave']  = spec3['wave']
    out3['obs_flux']  = spec3['flux']
    out3['obs_eflux'] = spec3['eflux']

    out['spec'] = [out1,out2,out3]

    # put in photometry
    out['phot'] = phot
    out['phot_filtarr'] = filtarr

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

    # define specNN which is a list of NN files assoicated with the input data spectra
    nspec = len(indict['data']['spec'])
    specNN = [specNN_rv31,specNN_rv31,specNN_rv31]

    print('---- Input Data ----')
    if 'phot' in indict['data'].keys():
        print('Phot:')
        for kk in indict['data']['phot'].keys():
            print('      {0} = {1:f} +/- {2:f}'.format(kk,*indict['data']['phot'][kk]))
    if 'spec' in indict['data'].keys():
        for ii in range(nspec):
            print('... For Demo Spec: {0} ...'.format(ii))
            spec_i = indict['data']['spec'][ii]
            print('number of pixels: {0}'.format(len(spec_i['obs_wave'])))
            print('min/max wavelengths: {0} -- {1}'.format(spec_i['obs_wave'].min(),spec_i['obs_wave'].max()))
            print('median flux: {0}'.format(np.median(spec_i['obs_flux'])))
            print('median flux error: {0}'.format(np.median(spec_i['obs_eflux'])))
            print('SNR: {0}'.format(np.median(spec_i['obs_flux']/spec_i['obs_eflux'])))
            print('      Npixels = {0}'.format(len(spec_i['obs_wave'])))
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
        'photjitter':1E-5,
        })
    
    # add initial guesses for spectral specific params
    for ii in range(nspec):
        initpars[f'vrad_{ii}'] = 0.0
        initpars[f'lsf_{ii}'] = 32000.0
        initpars[f'specjitter_{ii}'] = 1E-5
        initpars[f'pc0_{ii}'] = 1.0
        initpars[f'pc1_{ii}'] = 0.0
        initpars[f'pc2_{ii}'] = 0.0
        initpars[f'pc3_{ii}'] = 0.0

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
    indict['priors']['vstar'] = ['uniform',[0.0,250.0]]
    indict['priors']['vmic']  = ['uniform',[0.5,2.0]]

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[0.0,0.01,0.0,0.5]]
    indict['priors']['dist'] = ['uniform',[1.0,50.0]] # distance in pc
    indict['priors']['photjitter'] = ['fixed',0.0]

    for ii in range(nspec):
        # calibration priors
        indict['priors'][f'lsf_{ii}'] = ['tnormal',[30000.0,500.0,15000.0,40000.0]]
        indict['priors'][f'vrad_{ii}'] = ['uniform',[-5,5]]

        indict['priors'][f'pc0_{ii}'] = ['uniform',[0.8,1.2]]
        indict['priors'][f'pc1_{ii}'] = ['uniform',[-0.1,0.1]]
        indict['priors'][f'pc2_{ii}'] = ['uniform',[-0.1,0.1]]
        indict['priors'][f'pc3_{ii}'] = ['uniform',[-0.1,0.1]]

        # indict['priors'][f'specjitter_{ii}'] = ['fixed',0.0]
        indict['priors'][f'specjitter_{ii}'] = ['tnormal',[0.0,0.001,0.0,0.1]]

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

    # define specNN which is a list of NN files assoicated with the input data spectra
    nspec = len(indict['data']['spec'])
    specNN = [specNN_rv31,specNN_rv31,specNN_rv31]

    print('---- Input Data ----')
    if 'phot' in indict['data'].keys():
        print('Phot:')
        for kk in indict['data']['phot'].keys():
            print('      {0} = {1:f} +/- {2:f}'.format(kk,*indict['data']['phot'][kk]))
    if 'spec' in indict['data'].keys():
        for ii in range(nspec):
            print('... For Demo Spec: {0} ...'.format(ii))
            spec_i = indict['data']['spec'][ii]
            print('number of pixels: {0}'.format(len(spec_i['obs_wave'])))
            print('min/max wavelengths: {0} -- {1}'.format(spec_i['obs_wave'].min(),spec_i['obs_wave'].max()))
            print('median flux: {0}'.format(np.median(spec_i['obs_flux'])))
            print('median flux error: {0}'.format(np.median(spec_i['obs_eflux'])))
            print('SNR: {0}'.format(np.median(spec_i['obs_flux']/spec_i['obs_eflux'])))
            print('      Npixels = {0}'.format(len(spec_i['obs_wave'])))
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
        'photjitter':1E-5,
        })

    # add initial guesses for spectral specific params
    for ii in range(nspec):
        initpars[f'vrad_{ii}'] = 0.0
        initpars[f'lsf_{ii}'] = 32000.0
        initpars[f'specjitter_{ii}'] = 1E-5
        initpars[f'pc0_{ii}'] = 1.0
        initpars[f'pc1_{ii}'] = 0.0
        initpars[f'pc2_{ii}'] = 0.0
        initpars[f'pc3_{ii}'] = 0.0


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
    indict['priors']['initial_[Fe/H]'] = ['uniform',[-3.0,0.4]]
    indict['priors']['initial_[a/Fe]'] = ['uniform',[-0.15,0.55]]

    # spectra priors
    indict['priors']['vstar'] = ['uniform',[0.0,250.0]]
    indict['priors']['vmic']  = ['uniform',[0.5,2.0]]

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[0.0,0.01,0.0,0.5]]
    indict['priors']['dist'] = ['uniform',[1.0,50.0]] # distance in pc
    
    # indict['priors']['photjitter'] = ['uniform',[1E-6,1E-2]]
    indict['priors']['photjitter'] = ['fixed',0.0]

    for ii in range(nspec):
        # calibration priors
        indict['priors'][f'lsf_{ii}'] = ['tnormal',[30000.0,500.0,15000.0,40000.0]]
        indict['priors'][f'vrad_{ii}'] = ['uniform',[-5,5]]

        indict['priors'][f'pc0_{ii}'] = ['uniform',[0.8,1.2]]
        indict['priors'][f'pc1_{ii}'] = ['uniform',[-0.1,0.1]]
        indict['priors'][f'pc2_{ii}'] = ['uniform',[-0.1,0.1]]
        indict['priors'][f'pc3_{ii}'] = ['uniform',[-0.1,0.1]]

        # indict['priors'][f'specjitter_{ii}'] = ['fixed',0.0]
        indict['priors'][f'specjitter_{ii}'] = ['tnormal',[0.0,0.001,0.0,0.1]]


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
