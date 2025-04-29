import h5py
import glob
import numpy as np
from astropy.table import Table
from Payne.fitting.fitutils import airtovacuum
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy import constants
speedoflight = constants.c / 1000.0

def getdata(catfile=None,gaiaid=None,starind=None):
    # breakpoint()
    th5 = h5py.File(catfile,'r')
    cat_i = Table(th5['catalog'][()])

    if gaiaid != None:
        t = th5[f'{gaiaid}']
        # breakpoint()
        cat_ii = cat_i[cat_i['GAIAEDR3_ID'] == gaiaid]
        # breakpoint()
        cat = {x:cat_ii[x][0] for x in cat_ii.keys()}
        # breakpoint()

    elif starind != None:
        cat_ii = cat_i[starind]
        cat = {x:cat_ii[x] for x in cat_ii.keys()}
        
        gaiaid = cat['GAIAEDR3_ID']
        t = th5[f'{gaiaid}']
                
    header = {ii:th5['header'][ii][()] for ii in th5['header'].keys()}
    hdr_date = header['DATE-OBS']
    location = EarthLocation.of_site('MMT')

    sc = SkyCoord(ra=cat['GAIAEDR3_RA']*u.deg,dec=cat['GAIAEDR3_DEC']*u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(hdr_date), location=location)
    HC = float(barycorr.to(u.km/u.s).value)
    
    # Need to change the names of the Gaia EDR3 filters to DR3
    # These filters are unchanged between EDR3 and DR3
    #phot['GaiaDR3_BP'] = phot.pop('GaiaEDR3_BP')
    #phot['GaiaDR3_RP'] = phot.pop('GaiaEDR3_RP')
    #phot['GaiaDR3_G'] = phot.pop('GaiaEDR3_G')
    #t['phot']['GaiaDR3_BP'] = t['phot'].pop('GaiaEDR3_BP')
    #t['phot']['GaiaDR3_RP'] = t['phot'].pop('GaiaEDR3_RP')
    #t['phot']['GaiaDR3_G'] = t['phot'].pop('GaiaEDR3_G')

    out = {}
    out['spec'] = {}
    out['phot'] = {}
    
    # spec
    out['spec']['obs_wave']  = (1.0 - (HC/speedoflight)) * airtovacuum(t['wave'][()])
    out['spec']['obs_flux']  = t['flux'][()]
    out['spec']['obs_eflux'] = t['eflux'][()]

    # medflux = np.nanmedian(out['spec']['obs_flux'])
    # out['spec']['obs_flux'] = out['spec']['obs_flux'] / medflux
    # out['spec']['obs_eflux'] = out['spec']['obs_eflux'] / medflux

    SNR = np.nanmedian(out['spec']['obs_flux']/out['spec']['obs_eflux'])
    out['spec']['SNR'] = SNR
    out['spec']['date'] = hdr_date
    out['spec']['HC'] = HC

    # phot
    phot = t['phot']
    filterarr = list(phot.keys())
    usedfilters = []
    
    for ff in filterarr:            
        phot_i = phot[ff][()]
        if np.isfinite(phot_i[0]) & (phot_i[1] > 0.0):
            if 'Gaia' in ff:
                ff = ff.replace('EDR3', 'DR3')
            if 'PS_y' in ff:
                print(f"Skipping filter {ff}")
                continue
            out['phot'][ff] = [phot_i[0],phot_i[1]]
            usedfilters.append(ff)

    out['phot_filtarr'] = usedfilters
    
    # parallax
    out['parallax'] = [cat['GAIAEDR3_PARALLAX'],cat['GAIAEDR3_PARALLAX_ERROR']]
    
    # define some cluster guesses
    if 'ic348' in catfile:
        Avest = 1.91 # Cantat-Gaudin+ 2020
        RVest = 15.44 # Tarricq+ 2021
    elif 'm37' in catfile:
        Avest = 0.75
        RVest = 8.81
    elif 'm3' in catfile:
        Avest = 0.03 # Harris 1996
        RVest = -147.2 # BAUMGARDT H. and HILKER M 2018
    elif 'm67' in catfile:
        Avest = 0.07
        RVest = 34.18
    elif 'ngc6791' in catfile:
        Avest = 0.70
        RVest = -47.75
    elif 'ngc6811' in catfile:
        Avest = 0.09
        RVest = 7.17
    elif 'ngc6819' in catfile:
        Avest = 0.40
        RVest = 2.80
    elif 'ngc6866' in catfile:
        Avest = 0.48
        RVest = 12.44
    else:
        print('Could not understand cluster name in filename, setting Avest = 0.1 and RVest = 0.0')
        Avest = 0.1
        RVest = 0.0
    
    out['Avest'] = Avest
    out['RVest'] = RVest

    out['Teff']   = 5770.0
    out['[Fe/H]'] = 0.0
    out['[a/Fe]'] = 0.0
    out['log(g)'] = 4.5
    out['log(R)'] = 0.0

    return out

def getall(gaiaid=None,date=None,cluster=None):
    
    datasrcstr ='/data/labs/douglaslab/sofairj/data/hectochelle_rereduction/'
    
    filesrstr = '*'
    # list of all possible HDF5 files
    if cluster != None:
        filesrstr += f'{cluster}*'
    if date != None:
        filesrstr += f'{date}*'

    filesrstr += '.h5'
    flist = glob.glob(datasrcstr+filesrstr)

    # get the HDF5 files with this star
    workingfiles = []
    for ff in flist:
        with h5py.File(ff,'r') as th5:
            if f'{gaiaid}' in list(th5.keys()):
                workingfiles.append(ff)
    
    if len(workingfiles) == 0:
        print(f'Could Not Find Any Spectra for GaiaID = {gaiaid}')
        return None
    
    out = {}
    out['spec'] = []
    out['specname'] = []
    # breakpoint()
    for ii,ww in enumerate(workingfiles):
        # breakpoint()
        data = getdata(catfile=ww,gaiaid=gaiaid)
        if ii == 0:
            out['phot']     = data['phot']
            out['phot_filtarr'] = data['phot_filtarr']
            out['parallax'] = data['parallax']
            out['Avest']    = data['Avest']
            out['RVest']    = data['RVest']

            out['Teff']   = data['Teff']
            out['[Fe/H]'] = data['[Fe/H]']
            out['[a/Fe]'] = data['[a/Fe]']
            out['log(g)'] = data['log(g)']
            out['log(R)'] = data['log(R)']

        if data['spec']['SNR'] >= 3.0:
            out['spec'].append(data['spec'])
            out['specname'].append(ww.split('/')[-1])
    
    if len(out['spec']) > 2:
        # sort spectra by date
        sortind = np.argsort(out['specname'])
        out['spec'] = list(np.array(out['spec'])[sortind])
        out['specname'] = list(np.array(out['specname'])[sortind])

    return out
