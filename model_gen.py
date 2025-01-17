from Payne.fitting.genmod import GenMod
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


# set up paths for NN file and NN type

specNN = '/data/labs/douglaslab/sofairj/demo4/models/specNN/modV0_spec_LinNet_R42K_WL510_535_wvt.h5'

NNtype = 'LinNet'



# initialize model generation class

GM = GenMod()

GM._initspecnn(

    nnpath=specNN,

    NNtype=NNtype)        



# set the stellar parameters

starpars = ([

    5564.35, # Teff

    4.16, # log(g)

    -0.12, # [Fe/H]

    0.11, # [a/Fe]

    -98.13, # vrad

    2.74, # vstar

    1.0, # vmic

    32007.42, # lsf (R)

    1.08, # pc0

    0.06, # pc1

    0.0, # pc2

    0.0, # pc3

    ])



# generate the model, output is an array of [wave,flux]

specmod = GM.genspec(starpars,modpoly=True)

print(type(specmod))
print(specmod)
print(np.shape(specmod))

fig, ax = plt.subplots()

ax.plot(specmod[0], specmod[1])

plt.savefig("model_gen_plot.png")
