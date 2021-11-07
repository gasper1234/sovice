import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
import pickle

def tofile(sez, name):
	with open(name, 'wb') as pickle_file:
		pickle.dump(sez, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

def fromfile(name):
	with open(name, 'rb') as pickle_load:
		lst = pickle.load(pickle_load)
	return lst

def cor_fast(g):
	N = len(g)
	zeros = np.zeros(N)
	g = np.concatenate((g, zeros))
	g = fft.fft(g)
	g = g*np.conj(g)
	g = np.real(g)/2/N
	g = fft.ifft(g)
	g = fft.fftshift(g)
	#for i in range(1, N+1):
	#	g[i] /= i
	#for i in range(N+1, 2*N):
	#	g[i] /= 2*N - i
	g /= N
	return g


def ex_h_sq(a):
	h = sum(a)/len(a)
	return h**2

def cor(g, h):
	N = len(g)
	zeros = np.zeros(N)
	g_pad = np.concatenate((g, zeros))
	h_pad = np.concatenate((zeros, h))
	cor_val = np.zeros(2*N)
	for i in range(2*N):
		if i > N:
			cor_val[i] = np.dot(g_pad[:i+1], h_pad[2*N-i-1:])/(2*N-i)
		else:
			cor_val[i] = np.dot(g_pad[:i+1], h_pad[2*N-i-1:])/(i+1) #notaligned
	return cor_val

def cor_N(g, h):
	N = len(g)
	zeros = np.zeros(N)
	g_pad = np.concatenate((g, zeros))
	h_pad = np.concatenate((zeros, h))
	cor_val = np.zeros(2*N)
	for i in range(2*N):
		cor_val[i] = np.dot(g_pad[:i+1], h_pad[2*N-i-1:])/N
	return cor_val

def auto_cor(h, n):
	cor_0 = cor(0, h, h)
	cor_n = cor(n, h, h)
	h_sq = ex_h_sq(h)
	return (cor_n-h_sq)/(cor_0-h_sq)

def x_sq(a):
	return np.real(a)**2 + np.imag(a)**2

def x_abs(a):
	return np.sqrt(np.real(a)**2 + np.imag(a)**2)

def norm(g):
	M = max(g)
	g /= M
	return g