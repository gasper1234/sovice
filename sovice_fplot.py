import matplotlib.pyplot as plt
from sovice import *
from numpy import fft
import numpy as np
import re

txts = ['bubomono.txt', 'bubo2mono.txt', 'mix.txt', 'mix1.txt', 'mix2.txt', 'mix22.txt']
txts_name = ['sova 1', 'sova 2', 'črički', 'potok', 'črički + reka', 'reka']

fig, ax = plt.subplots(len(txts), 1, sharex=True)

for ind in range(len(txts)):

    with open(txts[ind]) as f:
        lines = f.readlines()
        data = [int(l.split('\n')[0]) for l in lines]

    N = len(data)

    data = data[:N]

    f_v = 44100

    dt = N/f_v
    N = len(data)
    delta_t = N/f_v

    t_1 = 0
    t = np.array([delta_t*i/N+t_1 for i in range(N)])
    t_ext = np.array([delta_t*i/N+t_1 for i in range(2*N)])

    f_c = f_v/2
    f = np.linspace(-f_c,f_c,N,endpoint=False)

    F_koef = fft.fft(data)/N # Fourier coefficients (divided by n)
    Fk = fft.fftshift(F_koef) # Shift zero freq to center

    Fk = norm(x_sq(Fk))
    ax[ind].plot(f, Fk, color='mediumblue')
    ax[ind].set_ylabel(r'$|H|^2$')
    #ax[ind].plot(f, cor_val, color='mediumblue')
    nu_disp = 700
    ax[ind].set_xlim(-nu_disp, nu_disp)
    ax[ind].set_title(txts_name[ind], rotation='vertical', x=1.02, y=0.1)
    if ind == 0:
        ax[ind].axvline(x=379,ymin=0,ymax=1,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False, label='sova 1')
        ax[ind].axvline(x=-380,ymin=0,ymax=1,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        continue
    elif ind == 1:
        ax[ind].axvline(x=379,ymin=0,ymax=1.2,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        ax[ind].axvline(x=-380,ymin=0,ymax=1.2,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        ax[ind].axvline(x=334,ymin=0,ymax=1,c="lime",linewidth=4, alpha=0.5, zorder=0,clip_on=False, label='sova 2')
        ax[ind].axvline(x=-335,ymin=0,ymax=1,c="lime",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
    else:
        ax[ind].axvline(x=379,ymin=0,ymax=1.2,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        ax[ind].axvline(x=-380,ymin=0,ymax=1.2,c="red",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        ax[ind].axvline(x=334,ymin=0,ymax=1.2,c="lime",linewidth=4, alpha=0.5, zorder=0,clip_on=False)
        ax[ind].axvline(x=-334,ymin=0,ymax=1.2,c="lime",linewidth=4, alpha=0.5, zorder=0,clip_on=False)        

ax[-1].set_xlabel(r'$\nu [Hz]$')
ax[0].legend()
ax[1].legend()
ax[2].legend(title='sova 1')
ax[3].legend(title='sova 1')
ax[4].legend(title='sova 1')
ax[5].legend(title='sova 2')
plt.show()