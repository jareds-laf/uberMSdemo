from astropy.table import Table
import numpy as np
import argparse
from jax import jit
import itertools

from Payne.jax.genmod import GenMod
import getdataH5

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

specNN = './models/specNN/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'
photNN = './models/photNN/'
NNtype = 'LinNet'
mistNN = './models/mistNN/mistyNN_2.3_v256_v0.h5'
SBlib  = './models/specNN/c3k_v1.3.sed_r500.h5'

def planck(wav, T):
    h = 6.626e-34
    c = 3.0e+8
    k = 1.38e-23

    wave_i = wav*(1E-10)
    a = 2.0*h*c**2
    b = h*c/(wave_i*k*T)
    intensity = a/ ( (wave_i**5) * (np.exp(b) - 1.0) )
    return intensity

from scipy import constants
speedoflight = constants.c / 1000.0

# useful constants
# speedoflight = 2.997924e+10
speedoflight_kms = 2.997924e+5
speedoflight_nms = 2.997924e+17
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
jansky_cgs = 1e-23
# value to go from L_sun to erg/s/cm^2 at 10pc
log_rsun_cgs = np.log10(6.955) + 10.0
log_lsun_cgs = np.log10(lsun)
log4pi = np.log10(4 * np.pi)

rng = np.random.default_rng()

def mkspec(ax_spec=None,ax_resid=None,
           waverange=None,
           mod=None,
           pmod=None,
           smod=None,
           data=None,
           labelx=True,
           labely=True):
    
    if waverange != None:
        # cond = (data['obs_wave'] >= waverange[0]-10.0) & (data['obs_wave'] <= waverange[1]+10)
        obs_wave  = data['obs_wave'] #[cond]
        obs_flux  = data['obs_flux'] #[cond]
        obs_eflux = data['obs_eflux'] #[cond]
    else:
        obs_wave  = data['obs_wave']
        obs_flux  = data['obs_flux']
        obs_eflux = data['obs_eflux']
        waverange = [obs_wave.min(),obs_wave.max()]
        
    ax_spec.plot(obs_wave,obs_flux,ls='-',lw=0.5,c='k',zorder=0)
    ax_spec.plot(mod[0],mod[1],ls='-',lw=1.0,c='C3',alpha=1.0,zorder=1)

    if pmod != None:
         ax_spec.plot(pmod[0],pmod[1],ls='-',lw=1.0,c='C0',alpha=0.5,zorder=1)
    if smod != None:
         ax_spec.plot(smod[0],smod[1],ls='-',lw=1.0,c='C1',alpha=0.5,zorder=1)
        

    if ax_resid != None:
        ax_resid.plot(obs_wave,
                    (mod[1]-obs_flux)/obs_eflux,
                    ls='-',lw=1.0,c='k',alpha=1.0)

    ax_spec.set_xlim(waverange[0],waverange[1])
    if labely:
        ax_spec.set_ylabel('Flux')

    if ax_resid != None:
        ax_resid.axhline(y=0.0, ls='-', lw=0.75,c='C3',alpha=0.85)
        ax_resid.axhline(y=-3.0,ls=':', lw=0.75,c='C3',alpha=0.85)
        ax_resid.axhline(y=3.0, ls=':', lw=0.75,c='C3',alpha=0.85)
        ax_resid.set_xlim(obs_wave.min(),obs_wave.max())
        if labelx:
            ax_resid.set_xlabel('Wavelength ['+r'$\AA$'+']')
        if labely:
            ax_resid.set_ylabel(r'$\chi$')
        ax_spec.set_xticklabels([])
    else:
        if labelx:
            ax_spec.set_xlabel('Wavelength ['+r'$\AA$'+']')

def mkphot(ax_phot=None,ax_flux=None,mod=None,data=None,bfdict=None):

    photdata = data

    # change dist back to pc
    dist = bfdict['dist'][0]*1000.0

    sedstr = (
        'GaiaDR3 G = {0:.2f}'.format(photdata['GaiaDR3_G'][0])
        )
    if 'PS_g' in photdata.keys():
        sedstr += '\n PS g = {0:.2f}'.format(photdata['PS_g'][0])
    if '2MASS_J' in photdata.keys():
        sedstr += '\n 2MASS J = {0:.2f}'.format(photdata['2MASS_J'][0])
    if 'WISE_W1' in photdata.keys():
        sedstr += '\n WISE W1 = {0:.2f}'.format(photdata['WISE_W1'][0])


    # ax_flux.text(0.97,0.97,sedstr,
    #     horizontalalignment='right',verticalalignment='top', 
    #     transform=ax_flux.transAxes,fontsize=8)

    from uberMS.utils import star_basis
    from uberMS.utils import photsys
    from uberMS.utils import ccm_curve

    SB = star_basis.StarBasis(
        libname=SBlib,
        use_params=['logt','logg','feh'],
        n_neighbors=1)

    WAVE_d = photsys.photsys()
    photbands_i = WAVE_d.keys()
    photbands = [x for x in photbands_i if x in photdata.keys()]
    WAVE = {pb:WAVE_d[pb][0] for pb in photbands}
    zeropts = {pb:WAVE_d[pb][2] for pb in photbands}
    fitsym = {pb:WAVE_d[pb][-2] for pb in photbands}
    fitcol = {pb:WAVE_d[pb][-1] for pb in photbands}
    filtercurves_i = photsys.filtercurves()
    filtercurves = {pb:filtercurves_i[pb] for pb in photbands}


    if bfdict['[Fe/H]_a'][0] >= 0.5:
        SEDfeh = 0.5
    elif bfdict['[Fe/H]_a'][0] <= -3.0:
        SEDfeh = -3.0
    else:
        SEDfeh = bfdict['[Fe/H]_a'][0]

    if bfdict['Teff_a'][0] <= 3500.0:
        SEDTeff = 3500.0
    else:
        SEDTeff = bfdict['Teff_a'][0]

    if bfdict['log(g)_a'][0] >= 5.5:
        SEDlogg = 5.5
    else:
        SEDlogg = bfdict['log(g)_a'][0]

    spec_w,spec_f,_ = SB.get_star_spectrum(
        logt=np.log10(SEDTeff),logg=SEDlogg,feh=SEDfeh)

    to_cgs_i = lsun/(4.0 * np.pi * (pc*dist)**2)
    nor = SB.normalize(logr=bfdict['log(R)_a'][0])*to_cgs_i
    spec_f = spec_f*nor
    spec_f = spec_f*(speedoflight/((spec_w*1E-8)**2.0))

    spec_f = np.nan_to_num(spec_f)
    spcond = spec_f > 1e-32
    spec_f = spec_f[spcond]
    spec_w = spec_w[spcond]
    
    extratio = ccm_curve.ccm_curve(spec_w/10.0,bfdict['Av'][0]/3.1)                    

    ax_flux.plot(spec_w/(1E+4),np.log10(spec_f/extratio),ls='-',lw=0.5,
        alpha=1.0,zorder=-1,c='C0')

    sedoutkeys = photdata.keys()
    modmag = [mod[kk] for kk in sedoutkeys]

    # split out data into phot and error dict
    initphot = {kk:photdata[kk][0] for kk in sedoutkeys if kk in photbands}
    initphoterr = {kk:photdata[kk][1] for kk in sedoutkeys if kk in photbands}

    obswave   = np.array([WAVE[kk] for kk in sedoutkeys])
    fitsym    = np.array([fitsym[kk] for kk in sedoutkeys])
    fitcol    = np.array([fitcol[kk] for kk in sedoutkeys])
    fc        = [filtercurves[kk] for kk in sedoutkeys]
    obsmag    = np.array([initphot[kk] for kk in sedoutkeys if kk in photbands])
    obsmagerr = np.array([initphoterr[kk] for kk in sedoutkeys if kk in photbands])
    # modmag    = np.array([initphot[kk] for kk in sedoutkeys if kk in photbands])
    # modmag    = np.array([sedout[kk] for kk in sedoutkeys])
    obsflux_i = np.array([zeropts[kk]*10.0**(initphot[kk]/-2.5) for kk in sedoutkeys if kk in photbands])
    obsflux   = [x*(jansky_cgs)*(speedoflight/((lamb*1E-8)**2.0)) for x,lamb in zip(obsflux_i,obswave)]
    modflux_i = np.array([zeropts[kk]*10.0**(x/-2.5) for x,kk in zip(modmag,sedoutkeys)])
    modflux   = [x*(jansky_cgs)*(speedoflight/((lamb*1E-8)**2.0)) for x,lamb in zip(modflux_i,obswave)]

    # plot the observed SED and MAGS
    minobsflx = np.inf
    maxobsflx = -np.inf
    for w,f,mod,s,clr in zip(obswave,obsflux,modflux,fitsym,fitcol):
        if np.log10(f) > -30.0:
            ax_flux.scatter(w/1E+4,np.log10(mod),marker=s,c='C3',zorder=1,s=5)
            ax_flux.scatter(w/1E+4,np.log10(f),marker=s,c='k',zorder=-1,s=20)
            if np.log10(f) < minobsflx:
                 minobsflx = np.log10(f)
            if np.log10(f) > maxobsflx:
                 maxobsflx = np.log10(f)

    for w,m,me,mod,s,clr in zip(obswave,obsmag,obsmagerr,modmag,fitsym,fitcol):
        # if np.abs(m-mod)/me > 5.0:
        #     me = np.abs(m-mod)
        if (m < 30) & (m > -30):
            ax_phot.scatter(w/1E+4,mod,marker='o',c='C3',zorder=1,s=5)
            ax_phot.errorbar(w/1E+4,m,yerr=me,ls='',marker=',',c='k',zorder=-1)
            ax_phot.scatter(w/1E+4,m,marker=s,c='k',zorder=-1,s=20)
    # ax_phot.axhline(y=0.0,c='C3',lw=1.0,ls='-',zorder=2,alpha=0.75)

    # plot filter curves
    for fc_i,clr in zip(fc,fitcol):
        trans_i = 0.25*fc_i['trans']*(0.9*maxobsflx-1.1*minobsflx)+1.1*minobsflx
        ax_flux.plot(fc_i['wave']/1E+4,trans_i,ls='-',lw=0.5,c=clr,alpha=1.0)

    ax_flux.set_ylim(1.1*minobsflx,0.9*maxobsflx)

    ax_flux.set_xlim([0.25,6.0])
    ax_flux.set_xscale('log')

    ax_phot.set_xlim([0.25,6.0])
    ax_phot.set_xscale('log')

    ax_phot.set_ylim(ax_phot.get_ylim()[::-1])
    # ax_phot.set_ylim(-0.1,0.1)

    ax_flux.set_xticks([0.3,0.5,0.7,1.0,3,5])
    ax_flux.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax_phot.set_xticks([0.3,0.5,0.7,1,3,5])
    ax_phot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax_flux.set_ylabel(r'log(F$_{\lambda}$) [erg s$^{-1}$ cm$^{-2}$]')

    ax_flux.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax_phot.set_ylabel(r'Mag.')

    ax_flux.yaxis.tick_right()
    ax_phot.yaxis.tick_right()
    ax_flux.yaxis.set_label_position('right')
    ax_phot.yaxis.set_label_position('right')
    # axSED.set_xticklabels([])

def runstar(gaiaid=None,sampdir=None,cluster=None,version='V0',mgtriplet=False,**kwargs):
    verbose = kwargs.get('verbose',False)

    # grab data
    data = getdataH5.getall(gaiaid=gaiaid, cluster=cluster)
    filtarray = data['phot_filtarr']

    specindex = kwargs.get('specindex',0)

    data['spec'] = data['spec'][specindex]

    # initialize prediction classes
    GM = GenMod()

    GM._initspecnn(
        nnpath=specNN,
        Cnnpath=None,
        NNtype=NNtype)
    GM._initphotnn(
        data['phot_filtarr'],
        nnpath=photNN)

    # pull out some information about NNs
    specNN_labels = GM.PP.modpars

    # jit a couple of functions
    genspecfn = GM.genspec
    genphotfn = GM.genphot

    # read in output file
    if sampdir is None:
        # set the output file name
        samplefile = f'./samples/samples_{version}.fits'
    else:
        # set the output file name
        samplefile = f'{sampdir}/samples_UTPbinary_{gaiaid}_{specindex}_{version}.fits'

    samples = Table.read(samplefile,format='fits')

    bfdict = {}
    for kk in samples.keys():
        bfdict[kk] = [np.median(samples[kk]),np.std(samples[kk])]

    ####
    
    # make the spectral prediciton
    specpars_p = ([
        bfdict['Teff_a'][0],bfdict['log(g)_a'][0],bfdict['[Fe/H]_a'][0],bfdict['[a/Fe]_a'][0],
        bfdict['vrad_a'][0],bfdict['vstar_a'][0],bfdict['vmic_a'][0],bfdict['lsf'][0]])
    specpars_p += [bfdict[f'pc0'][0],bfdict[f'pc1'][0],bfdict[f'pc2'][0],bfdict[f'pc3'][0]]
    specmod_p = genspecfn(specpars_p,outwave=data['spec']['obs_wave'],modpoly=True)
    specmod_p = np.array(specmod_p[1])

    specpars_pn = ([
        bfdict['Teff_a'][0],bfdict['log(g)_a'][0],bfdict['[Fe/H]_a'][0],bfdict['[a/Fe]_a'][0],
        bfdict['vrad_a'][0],bfdict['vstar_a'][0],bfdict['vmic_a'][0],bfdict['lsf'][0]])
    specpars_pn += [1.0,0.0]
    specmod_pn = genspecfn(specpars_pn,outwave=data['spec']['obs_wave'],modpoly=True)
    specmod_pn = np.array(specmod_pn[1])


    specpars_s = ([
        bfdict['Teff_b'][0],bfdict['log(g)_b'][0],bfdict['[Fe/H]_b'][0],bfdict['[a/Fe]_b'][0],
        bfdict['vrad_b'][0],bfdict['vstar_b'][0],bfdict['vmic_b'][0],bfdict['lsf'][0]])
    specpars_s += [1.0,0.0]
    specmod_s = genspecfn(specpars_s,outwave=data['spec']['obs_wave'],modpoly=True)
    specmod_s = np.array(specmod_s[1])

    radius_a = 10.0**bfdict['log(R)_a'][0]
    radius_b = 10.0**bfdict['log(R)_b'][0]

    R = (
        (planck(data['spec']['obs_wave'],bfdict['Teff_a'][0]) * radius_a**2.0) / 
        (planck(data['spec']['obs_wave'],bfdict['Teff_b'][0]) * radius_b**2.0)
         )
    specmod_est = (specmod_p + R * specmod_s) / (1.0 + R)

    specmod_p  = specmod_p * R / (1.0 + R)
    specmod_pn = specmod_pn / (1.0 + R)
    specmod_s  = specmod_s * R / (1.0 + R)

    # make photometry prediction
    photpars_p = ([
        bfdict['Teff_a'][0],bfdict['log(g)_a'][0],bfdict['[Fe/H]_a'][0],bfdict['[a/Fe]_a'][0],
        bfdict['log(R)_a'][0],bfdict['dist'][0],bfdict['Av'][0],3.1])
    photmod_p = genphotfn(photpars_p)
    photmod_p = [photmod_p[xx] for xx in filtarray]

    photpars_s = ([
        bfdict['Teff_b'][0],bfdict['log(g)_b'][0],bfdict['[Fe/H]_b'][0],bfdict['[a/Fe]_b'][0],
        bfdict['log(R)_b'][0],bfdict['dist'][0],bfdict['Av'][0],3.1])
    photmod_s = genphotfn(photpars_s)
    photmod_s = [photmod_s[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * np.log10( 10.0**(-0.4 * m_p) + 10.0**(-0.4 * m_s) )
         for m_p,m_s in zip(photmod_p,photmod_s)
         ] 
    )
    photmod_est = {x:photmod_est[ii] for ii,x in enumerate(filtarray)}
    
    parstr = 'GaiaDR3 ID = {}\n'.format(gaiaid)
    parstr += r'GaiaDR3 $\pi$' + ' = {0:.3f} +/- {1:.3f} mas\n'.format(*data['parallax'])
    parstr += '--- Primary --- \n'
    parstr += r'T$_{eff}$' + ' = {0:.0f} +/- {1:.0f} K\n'.format(*bfdict['Teff_a'])
    parstr += r'log(g)'      + ' = {0:.3f} +/- {1:.3f} \n'.format(*bfdict['log(g)_a'])
    parstr += r'V$_{mic}$'   + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vmic_a'])
    parstr += r'V$_{\bigstar}$' + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vstar_a'])
    parstr += r'V$_{RV}$' + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vrad_a'])
    parstr += '--- Secondary --- \n'
    parstr += r'T$_{eff}$' + ' = {0:.0f} +/- {1:.0f} K\n'.format(*bfdict['Teff_b'])
    parstr += r'log(g)'      + ' = {0:.3f} +/- {1:.3f} \n'.format(*bfdict['log(g)_b'])
    parstr += r'V$_{mic}$'   + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vmic_b'])
    parstr += r'V$_{\bigstar}$' + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vstar_b'])
    parstr += r'V$_{RV}$' + ' = {0:.3f} +/- {1:.3f} km/s\n'.format(*bfdict['vrad_b'])
    parstr += '--- System --- \n'
    parstr += r'[Fe/H]'      + ' = {0:.3f} +/- {1:.3f} \n'.format(*bfdict['[Fe/H]_a'])
    parstr += r'[a/Fe]'      + ' = {0:.3f} +/- {1:.3f} \n'.format(*bfdict['[a/Fe]_a'])
    parstr += r'Dist'        + ' = {0:.1f} +/- {1:.1f} pc \n'.format(*bfdict['dist'])
    parstr += r'A$_{V}$'     + ' = {0:.3f} +/- {1:.3f} \n'.format(*bfdict['Av'])

    # define output file
    outfile = f'./plots/compmod_UTPcomparesmes_{gaiaid}_{specindex}_{version}.pdf'

    with PdfPages(outfile) as pdf:

        ##### Make model comparison plot #####

        fig = plt.figure(figsize=(10,8))#,constrained_layout=True)
        gs = gridspec.GridSpec(6, 6, figure=fig)
        gs.update(hspace=0.05)

        ax_main_spec  = fig.add_subplot(gs[:3,:-2])
        ax_main_resid = fig.add_subplot(gs[3:4,:-2])

        # ax_reg1_spec = fig.add_subplot(gs[4:,:2])
        # ax_reg2_spec = fig.add_subplot(gs[4:,2:4])


        if mgtriplet:
            mkspec(ax_spec=ax_main_spec,ax_resid=ax_main_resid,
                   waverange=[np.amin(data['spec']['obs_wave']),5190],mod=[data['spec']['obs_wave'],specmod_est],
                   pmod=[data['spec']['obs_wave'],specmod_pn],
                   smod=[data['spec']['obs_wave'],specmod_s],
                   data=data['spec'],labelx=True)
        else:
            mkspec(ax_spec=ax_main_spec,ax_resid=ax_main_resid,
                   waverange=None,mod=[data['spec']['obs_wave'],specmod_est],
                   pmod=[data['spec']['obs_wave'],specmod_pn],
                   smod=[data['spec']['obs_wave'],specmod_s],
                   data=data['spec'],labelx=True)

        # mkspec(ax_spec=ax_reg2_spec,ax_resid=None,
        #        waverange=[5260,5272],mod=[data['spec']['obs_wave'],specmod_est],
        #        pmod=[data['spec']['obs_wave'],specmod_pn],
        #        smod=[data['spec']['obs_wave'],specmod_s],
        #        data=data['spec'],labelx=False,labely=False)

        ax_main_phot = fig.add_subplot(gs[:2,-2:])
        ax_main_flux = fig.add_subplot(gs[2:4,-2:])
        
        mkphot(ax_phot=ax_main_phot,ax_flux=ax_main_flux,
               mod=photmod_est,data=data['phot'],bfdict=bfdict)
        
        fig.align_labels()
 
        pdf.savefig(fig)
        plt.close(fig)

        # ##### Make corner plot  ########
        
        # change distance into kpc
        samples['dist'] = samples['dist']/1000.0

        # recalculate bf for dist and vrad
        bfdict['dist'] = [np.nanmedian(samples['dist']),np.nanstd(samples['dist'])]
        
        # if parallax isn't in samples, create it
        if 'parallax' not in samples.keys():
            samples['parallax'] = 1.0/samples['dist']
            bfdict['parallax'] = [np.nanmedian(samples['parallax']),np.nanstd(samples['parallax'])]

        # list of parameters to include in corner plot        
        # pltpars_i = [
        #     'Teff_a','log(g)_a','[Fe/H]_a','[a/Fe]_a','vmic_a','vstar_a',
        #     'Teff_b','log(g)_b','vmic_b','vstar_b','specjitter',
        #     'photjitter','dist','parallax','log(R)_a','log(R)_b','Av',
        #     ]
        # pltpars_i = [
        #     'Teff_a','log(g)_a','Teff_b','log(g)_b',
        #     'vstar_a','vstar_b','[Fe/H]_a',
        #     ]
        pltpars_i = [
            'Teff_a','log(g)_a','[Fe/H]_a','[a/Fe]_a','vmic_a','vstar_a','vrad_a',
            'Teff_b','log(g)_b','vmic_b','vstar_b','vrad_b',
            'lsf','pc0','pc1','pc2','pc3','specjitter',
            'photjitter','dist','parallax','log(R)_a','log(R)_b','Av',
            ]

        # check to see if any of these parameters have been fixed
        pltpars = []
        for pp in pltpars_i:
            if samples[pp].min() != samples[pp].max():
                pltpars.append(pp)
        pltpars = np.array(pltpars)

        parind = np.array(range(len(pltpars)))

        gaia_parallax = data['parallax']

        fig = plt.figure(figsize=(20,20))
        # fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(len(pltpars),len(pltpars))
        gs.update(wspace=0.15,hspace=0.15)

        fig.text(
            0.75, 0.35, 
            parstr,
            fontsize = 10,
        )

        nbins = 35

        for kk in itertools.product(pltpars,pltpars):
            kkind1 = parind[pltpars == kk[0]][0]
            kkind2 = parind[pltpars == kk[1]][0]
            ax = fig.add_subplot(gs[kkind1,kkind2])

            if kkind1 < kkind2:
                ax.set_axis_off()
                continue
            
            Xminran = bfdict[kk[0]][0] - 5.0 * bfdict[kk[0]][1]
            Xmaxran = bfdict[kk[0]][0] + 5.0 * bfdict[kk[0]][1]
            Yminran = bfdict[kk[1]][0] - 5.0 * bfdict[kk[1]][1]
            Ymaxran = bfdict[kk[1]][0] + 5.0 * bfdict[kk[1]][1]

            if kk[0] == 'vstar':
                Xminran = max([0.0,Xminran])
            if kk[1] == 'vstar':
                Yminran = max([0.0,Yminran])

            if kk[0] == '[Fe/H]':
                Xminran = max([-4.0,Xminran])
                Xmaxran = min([0.5,Xmaxran])
            if kk[1] == '[Fe/H]':
                Yminran = max([-4.0,Yminran])
                Ymaxran = min([0.5,Ymaxran])

            if kk[0] == '[a/Fe]':
                Xminran = max([-0.2,Xminran])
                Xmaxran = min([0.6,Xmaxran])
            if kk[1] == '[a/Fe]':
                Yminran = max([-0.2,Yminran])
                Ymaxran = min([0.6,Ymaxran])

            if kk[0] == 'Av':
                Xminran = max([0.0,Xminran])
            if kk[1] == 'Av':
                Yminran = max([0.0,Yminran])

            if kk[0] == 'specjitter':
                Xminran = max([0.0,Xminran])
            if kk[1] == 'specjitter':
                Yminran = max([0.0,Yminran])

            if kk[0] == 'photjitter':
                Xminran = max([0.0,Xminran])
            if kk[1] == 'photjitter':
                Yminran = max([0.0,Yminran])

            if kk[0] == 'dist':
                Xminran = max([0.0,Xminran])
            if kk[1] == 'dist':
                Yminran = max([0.0,Yminran])

            if kk[0] == 'vmic':
                Xminran = max([0.5,Xminran])
                Xmaxran = min([2.5,Xmaxran])
            if kk[1] == 'vmic':
                Yminran = max([0.5,Yminran])
                Ymaxran = min([2.5,Ymaxran]) 

            if kk[0] == 'log(g)':
                Xminran = max([0.0,Xminran])
                Xmaxran = min([5.5,Xmaxran])
            if kk[1] == 'log(g)':
                Yminran = max([0.0,Yminran])
                Ymaxran = min([5.5,Ymaxran]) 

            xarr_range = [Xminran,Xmaxran]
            yarr_range = [Yminran,Ymaxran]

            if kk[0] == kk[1]:

                n,bins,_ = ax.hist(
                samples[kk[0]],
                bins=nbins,
                histtype='step',
                linewidth=1.5,
                density=True,
                range=xarr_range,
                )

                # if (kk[0] == 'pc_0'):
                #     xarr = np.linspace(0.5,2.0,200)
                #     yarr = n.max()*np.ones(len(xarr))
                #     ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                # pcterms = {'pc_1':[0.0,0.5],'pc_2':[0.0,0.5],'pc_3':[0.0,0.25]}
                # if kk[0] in ['pc_1','pc_2','pc_3']:
                #     # xarr = np.linspace(xarr_range[0],xarr_range[1],200)
                #     xarr = np.linspace(-0.5,0.5,200)
                #     yarr = np.exp( -0.5*((xarr-pcterms[kk[0]][0])**2.0)/(pcterms[kk[0]][1]**2.0))
                #     yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                #     ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                if (kk[0] == 'parallax'):
                    xarr = np.linspace(xarr_range[0],xarr_range[1],500)
                    # xarr = np.logspace(-4,1,500)
                    yarr = np.exp( -0.5*((xarr-gaia_parallax[0])**2.0)/(gaia_parallax[1]**2.0) )
                    yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                    ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                # if (kk[0] == 'Vrot'):
                #     xarr = np.logspace(0,250,500)
                #     a = 1.05
                #     b = 1.5
                #     loc = 0.0
                #     scale = 250.0
                #     yarr = beta.pdf(xarr,a,b,loc=loc,scale=scale)
                #     yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                #     ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                # if (kk[0] == 'dist'):
                #     if gaia_parallax[0] > 0.0:
                #         if 3.0*gaia_parallax[1] < gaia_parallax[0]:
                #             maxdist = 1.0/(gaia_parallax[0]-3.0*gaia_parallax[1])
                #         else:
                #             maxdist = 200.0

                #         mindist = 1.0/(gaia_parallax[0]+3.0*gaia_parallax[1])
                #     else:
                #         mindist = max([1.0,1.0/(3.0*gaia_parallax[1])])
                #         maxdist = 200.0

                #     # xarr = np.linspace(self.SAMPLEStab['Dist'].min(),self.SAMPLEStab['Dist'].max(),200)
                #     xarr = np.linspace(mindist,maxdist,200)
                #     yarr = np.exp(AP.gal_lnprior(xarr,coords=lb_coords))
        
                #     if isinstance(yarr,float):
                #         yarr = 0.0*np.ones_like(xarr)
                #     else:
                #         yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                #     ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                ax.set_xlim(xarr_range[0],xarr_range[1])
                ylimtmp = ax.get_ylim()
                ax.set_ylim(ylimtmp[0],1.25*ylimtmp[1])
                ax.set_yticks([])
                if kk[0] != pltpars[-1]:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(kk[0])

                if 'Teff' in kk[0]:
                    ax.text(
                        0.5,1.1,
                        '{0:.0f} +/- {1:.0f}'.format(*bfdict[kk[0]]),
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=ax.transAxes,
                        fontsize=10,
                    )
                else:
                    ax.text(
                        0.5,1.1,
                        '{0:.3f} +/- {1:.2f}'.format(*bfdict[kk[0]]),
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=ax.transAxes,
                        fontsize=8,
                    )


            else:
                ax.hist2d(
                    samples[kk[1]],
                    samples[kk[0]],
                    bins=nbins,
                    cmap='Blues',
                    range=[yarr_range,xarr_range],
                    )

                ax.set_xlim(yarr_range[0],yarr_range[1])
                ax.set_ylim(xarr_range[0],xarr_range[1])

            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            [l.set_fontsize(6) for l in ax.get_xticklabels()]
            [l.set_fontsize(6) for l in ax.get_yticklabels()]

            labelcol = 'k'

            try:
                isfc = ax.get_subplotspec().is_first_col()
                islc = ax.get_subplotspec().is_last_col()
                isfr = ax.get_subplotspec().is_first_row()
                islr = ax.get_subplotspec().is_last_row()
            except AttributeError:
                isfc = ax.is_first_col()
                islc = ax.is_last_col()
                isfr = ax.is_first_row()
                islr = ax.is_last_row()

            if not isfc:
                ax.set_yticks([])
            elif isfc & isfr:
                ax.set_yticks([])
            elif kk[0] == pltpars[0]:
                pass
            else:
                if kk[0] == '[a/Fe]':
                    ax.set_ylabel('['+r'$\alpha$'+'/Fe]')
                elif kk[0] == '[Fe/H]_a':
                    ax.set_ylabel('[Fe/H]')
                elif kk[0] == 'Teff':
                    ax.set_xlabel(r'T$_{eff}$')
                elif kk[0] == 'vrad':
                    ax.set_ylabel(r'V$_{rad}$')
                elif kk[0] == 'vmic':
                    ax.set_ylabel(r'V$_{mic}$')
                elif kk[0] == 'vstar_a':
                    ax.set_ylabel(r'V$_{\bigstar}$'+'_a')
                elif kk[0] == 'vstar_b':
                    ax.set_ylabel(r'V$_{\bigstar}$'+'_b')
                elif kk[0] == 'parallax':
                    ax.set_ylabel(r'$\pi$')
                elif 'pc' in kk[0]:
                    ax.set_ylabel(r'pc$_{0}$'.format(kk[0].replace('pc','')))
                elif kk[0] == 'photjitter':
                    ax.set_ylabel(r'$\epsilon_{phot}$')
                elif kk[0] == 'specjitter':
                    ax.set_ylabel(r'$\epsilon_{spec}$')
                elif kk[0] == 'lsf':
                    ax.set_ylabel('LSF')
                elif 'Av' == kk[0]:
                    ax.set_ylabel(r'A$_{V}$')
                else:
                    ax.set_ylabel(kk[0])

            if not islr:
                ax.set_xticks([])
            else:
                if kk[1] == '[a/Fe]':
                    ax.set_xlabel('['+r'$\alpha$'+'/Fe]')
                elif kk[1] == '[Fe/H]_a':
                    ax.set_xlabel('[Fe/H]')
                elif kk[1] == 'Teff':
                    ax.set_xlabel(r'T$_{eff}$'+'\n[K]')
                elif kk[1] == 'dist':
                    ax.set_xlabel('Dist.'+'\n[kpc]')
                elif kk[1] == 'vrad_a':
                    ax.set_xlabel(r'V$_{rad,A}$'+'\n[km/s]')
                elif kk[1] == 'vrad_b':
                    ax.set_xlabel(r'V$_{rad,B}$'+'\n[km/s]')
                elif kk[1] == 'vmic':
                    ax.set_xlabel(r'V$_{mic}$'+'\n[km/s]')
                elif kk[1] == 'vstar_a':
                    ax.set_xlabel(r'V$_{\bigstar}$'+'_a'+'\n[km/s]')
                elif kk[1] == 'vstar_b':
                    ax.set_xlabel(r'V$_{\bigstar}$'+'_b'+'\n[km/s]')
                elif kk[1] == 'parallax':
                    ax.set_xlabel(r'$\pi$'+'\n["]')
                elif kk[1] == 'photjitter':
                    ax.set_xlabel(r'$\epsilon_{phot}$')
                elif kk[1] == 'specjitter':
                    ax.set_xlabel(r'$\epsilon_{spec}$')
                elif kk[1] == 'lsf':
                    ax.set_xlabel('LSF \n[x 1000]')
                elif 'pc' in kk[1]:
                    ax.set_xlabel(r'pc$_{0}$'.format(kk[1].replace('pc','')))
                elif 'Av' == kk[1]:
                    ax.set_xlabel(r'A$_{V}$')
                else:
                    ax.set_xlabel(kk[1])

        fig.align_labels()
        
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampdir',help='path for directory where samples are stored',type=str,default=None)
    parser.add_argument('--version','-v',help='analysis run version number',type=str,default='V0')
    parser.add_argument('--cluster',help='name of binary host cluster',type=str,default=None)
    parser.add_argument('--gaiaid','-i',help='gaiaid of binary',type=int, default=None)
    parser.add_argument('--specindex','-si',help='index of spectrum', type=int,default=None)
    parser.add_argument('--mgtriplet',help='focus spectrum around Mg I triplet',dest='mgtriplet',action='store_true',default=False)

    args = parser.parse_args()
    runstar(**vars(args))    
