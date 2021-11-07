import matplotlib.pyplot as plt
from sovice import *
from numpy import fft
import numpy as np
import re

txts = ['bubomono.txt', 'bubo2mono.txt', 'mix.txt', 'mix1.txt', 'mix2.txt', 'mix22.txt']
txts_name = ['sova 1', 'sova 2', 'črički', 'potok', 'črički + reka', 'reka']

fig, ax = plt.subplots(len(txts), 1, constrained_layout=True)

for ind in range(len(txts)):


    with open(txts[ind]) as f:
        lines = f.readlines()
        data = [int(l.split('\n')[0]) for l in lines]

    N = len(data)

    data = data[:N]


#autocorrelation

    f_v = 44100

    dt = N/f_v
    N = len(data)
    delta_t = N/f_v

    t_1 = 0
    t = np.array([delta_t*i/N+t_1 for i in range(N)])
    t_ext = np.array([delta_t*i/N+t_1 for i in range(2*N)])

#test function

    cor_val = cor_fast(data)


    f_c = f_v/2
    f = np.linspace(-f_c,f_c,N,endpoint=False)
    f_cor = np.linspace(-f_c, f_c, 2*N, endpoint=False)

    F_koef = fft.fft(data)/N # Fourier coefficients (divided by n)
    Fk = fft.fftshift(F_koef) # Shift zero freq to center

    F_koef_cor = fft.fft(cor_val)/N
    Fk_cor = fft.fftshift(F_koef_cor)

    #ax[0].plot(t, data)
    ax[ind].plot(t_ext, cor_val, color='mediumblue')
    ax[ind].set_xlim(5, 10)
    #ax[ind].set_ylim(-1.2*max(cor_val[N-100:N+100]), 1.2*max(cor_val[N-100:N+100]))
    ax[ind].set_ylim(-10, 10)
    ax[ind].set_title(txts_name[ind], rotation='vertical', x=1.02, y=0.1)
    #ax[ind].set_xlabel('t', x=0.55, y=0.03)
    #ax[2].plot(f_cor, x_sq(Fk_cor), color='purple')
    nu_disp = 2000
    #ax[2].set_xlim(-1*nu_disp, nu_disp)
    #ax[3].plot(f, x_sq(Fk))
    #ax[3].set_xlim(-1*nu_disp, nu_disp)

plt.show()
'''
def f(x):
    return np.sin(x)+np.sin(x/2)

N = 100
data_x = np.linspace(0, np.pi*18, N)
data_x_ex = np.linspace(0, np.pi*18*2, N*2)

data = f(data_x)


print(data)
#correlation

cor_val = cor_fast(data)

cor_val_1 = cor(data, data)

plt.plot(data_x, data)
plt.plot(data_x_ex, cor_val, color='purple')
plt.plot(data_x_ex, cor_val_1, color='pink')
plt.show()
'''


#save slow correlation to files
'''
try:
    cor_val = fromfile('cor_1'+txts[ind])
except:
    cor_val = cor(data, data)
    tofile(cor_val, 'cor_1'+txts[ind])
'''

#fourie
