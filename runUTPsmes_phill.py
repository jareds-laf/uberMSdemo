from uberMS.smes import runSVI
import numpy as np
import argparse
from astropy.table import Table
import getdataH5

specNN_rv31  = '/Users/pcargile/Astro/ThePayne/specANN/rv31/v256/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'
photNN = '/Users/pcargile/Astro/ThePayne/photANN/'
NNtype = 'LinNet'

def runTP(gaiaid=None,dospec=True,dophot=True,outputname=None,progressbar=True,version='V0',**kwargs):
    if gaiaid == None:
        print('user did not give a Gaia ID')
        return

    if (dospec == False) & (dophot==False):
        print('User did not set either dospec and/or dophot, returning nothing')
        return 

    # grab data
    data = getdataH5.getall(gaiaid=gaiaid)

    specindex = kwargs.get('specindex',0)

    data['spec'] = data['spec'][specindex]

    # init input dictionary
    indict = {}
    
    if outputname is None:
        # set the output file name
        indict['outfile'] = f'./samples/samples_{gaiaid}_UTPsmes_{version}.fits'
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
    
    teffest  = data['Teff']
    fehest   = data['[Fe/H]']
    afeest   = data['[a/Fe]']
    loggest  = data['log(g)']
    logRest  = data['log(R)']
    Avest    = data['Avest']
    RVest    = data['RVest']

    initpars = ({
        'Teff_p':5770.0,
        'Teff_s':5000.0,
        'log(g)_p':4.0,
        'log(g)_s':5.0,
        '[Fe/H]_p':0.0,
        '[Fe/H]_s':0.0,
        '[a/Fe]_p':0.0,
        '[a/Fe]_s':0.0,
        'log(R)_p':0.0,
        'log(R)_s':0.0,
        'vrad_p':-10.0,
        'vrad_s':10.0,
        'vstar_p':1.0,
        'vstar_s':1.0,
        'vmic_p':1.0,
        'vmic_s':1.0,
        'dist':distest,
        'Av':Avest,
        'specjitter':1E-5,
        'photjitter':1E-5,
        })
    
 
    indict['initpars'] = initpars

    print('------ Init Parameters ---')
    for kk in initpars.keys():
        print('      {0} = {1}'.format(kk,initpars[kk]))

    # define priors
    indict['priors'] = {}

    for kk in ['p','s']:
        # stellar priors
        indict['priors'][f'Teff_{kk}']    = ['uniform',[2500.0,10000.0]]
        indict['priors'][f'log(g)_{kk}']  = ['uniform',[0.0,5.5]]
        # indict['priors'][f'[Fe/H]_{kk}']  = ['uniform',[-3.0,0.5]]
        # indict['priors'][f'[a/Fe]_{kk}']  = ['uniform',[-0.2,0.6]]
        indict['priors'][f'log(R)_{kk}']  = ['uniform',[-3,3]]

        # spectra priors
        # indict['priors'][f'vstar_{kk}'] = ['uniform',[0.0,250.0]]
        indict['priors'][f'vstar_{kk}'] = ['tnormal',[0.0,4.0,0.0,50.0]]
        indict['priors'][f'vmic_{kk}']  = ['uniform',[0.5,2.0]]
        indict['priors'][f'vrad_{kk}']  = ['uniform',[RVest-100.0,RVest+100.0]]
        # if kk == 'p':
        #     indict['priors'][f'vrad_{kk}']  = ['uniform',[-50.0,50.0]]
        # else:
        #     indict['priors'][f'vrad_{kk}']  = ['uniform',[-50.0,50.0]]

    # fix chemistry to be identical
    indict['priors']['binchem'] = ['binchem','fixed']

    # photometry priors
    indict['priors']['Av'] = ['tnormal',[Avest,0.1,0.0,Avest+1.0]]
    indict['priors']['dist'] = ['uniform',[distmin,distmax]] # distance in pc

    indict['priors']['lsf'] = ['tnormal',[32000.0,100.0,15000.0,40000.0]]

    indict['priors']['pc0'] = ['uniform',[0.85,1.05]]
    # indict['priors']['pc0'] = ['fixed',1.0]
    indict['priors']['pc1'] = ['uniform',[-0.25,0.25]]
    indict['priors']['pc2'] = ['uniform',[-0.1,0.1]]
    indict['priors']['pc3'] = ['uniform',[-0.05,0.05]]

    # indict['priors']['specjitter'] = ['fixed',0.0]
    indict['priors']['specjitter'] = ['tnormal',[0.0,0.001,0.0,0.01]]
    # indict['priors']['photjitter'] = ['fixed',0.0]
    indict['priors']['photjitter'] = ['tnormal',[0.0,0.001,0.0,0.01]]

    print('------ Priors -----')
    for kk in indict['priors'].keys():    
        print('       {0}: {1}'.format(kk,indict['priors'][kk]))

    # define SVI parameters
    indict['svi'] = ({
        'steps':10000,
        'opt_tol':1E-6,
        'start_tol':1E-2,
        'progress_bar':progressbar,
        'post_resample':30000,
        })

    print('... Running TP')
    SVI = runSVI.sviTP(specNN=specNN_rv31,photNN=photNN,NNtype=NNtype,verbose=True)
    SVI.run(indict)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaiaid','-i',dest='gaiaid',help='gaia id for star',default=None,type=int)
    parser.add_argument('--specindex','-si',dest='specindex',help='index of spectrum for given star',default=0,type=int)
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
