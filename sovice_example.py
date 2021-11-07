import matplotlib.pyplot as plt
from sovice import *
from numpy import fft
import numpy as np
import re

txts = ['bubomono.txt', 'bubo2mono.txt', 'mix.txt', 'mix1.txt', 'mix2.txt', 'mix22.txt']
txts_name = ['sova 1', 'sova 2', 'črički', 'potok', 'črički + reka', 'reka']

ind = 1

fig, ax = plt.subplots(4, 1)

with open(txts[ind]) as f:
	lines = f.readlines()
	data = [int(l.split('\n')[0]) for l in lines]

N = len(data)

cor_val = cor_fast(data)

f_v = 44100

dt = N/f_v
N = len(data)
delta_t = N/f_v

t_1 = 0
t = np.array([delta_t*i/N+t_1 for i in range(N)])
t_ext = np.array([delta_t*i/N+t_1 for i in range(2*N)])

f_c = f_v/2
f = np.linspace(-f_c,f_c,N,endpoint=False)
f_cor = np.linspace(-f_c, f_c, 2*N, endpoint=False)


F_koef = fft.fft(data)/N # Fourier coefficients (divided by n)
Fk = fft.fftshift(F_koef) # Shift zero freq to center

F_koef_cor = fft.fft(cor_val)/N
Fk_cor = fft.fftshift(F_koef_cor)

nu_disp = 1000
ax[0].plot(t, data, color='mediumblue', label='zvok')
ax[0].set_xlim(-0.1, 10.1)
plt.setp(ax[0].get_xticklabels(), visible=False)
ax[0].set_ylabel(r'h')
ax[0].legend()
ax[1].plot(t_ext, cor_val, color='red', label='avtokorelacija')
ax[1].set_xlabel('t[s]', size=12)
ax[1].xaxis.set_label_coords(1.05, 0.1)
ax[1].set_ylabel(r'$\phi_{hh}$')
ax[1].legend()
ax[2].plot(f, norm(x_abs(Fk)), color='purple', label='FT zvok')
ax[2].set_xlim(-nu_disp, nu_disp)
plt.setp(ax[2].get_xticklabels(), visible=False)
ax[2].set_ylabel(r'$H$')
ax[2].legend()
ax[3].plot(f_cor, norm(x_abs(Fk_cor)), color='purple', label='FT avtokorelacija')
ax[3].set_xlim(-nu_disp, nu_disp)
ax[3].set_xlabel(r'$\nu [Hz]$', size=12)
ax[3].xaxis.set_label_coords(1.05, 0.1)
ax[3].set_ylabel(r'$H_{kor}$')
ax[3].legend()

plt.show()