

#Python version to check a GPU integration

#To render
from django.shortcuts import render,redirect

#To getroute
from . import a_getroute

#Numpy
import numpy as np

#Numba to GPU operations
from numba import cuda , types

#Functionality from time library
import time

#To debug
import pdb

#For sinus
import math 

import matplotlib.pyplot as plt # for scatter plot


def GPU_testGama(request):
	#To measure time	
	start = time.perf_counter()
	#testIntegration()
	#Integration in CPU
	testIntegrationDrGama()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')

	return render(request,'showmap.html')

def testIntegrationDrGama():
	
	#First with CPU
	z0 = np.array([2,3])
	tspan = np.array([0,10])
	k1 = 0
	k2 = 1
	sample = .001
	tt,zz = odeFelCPU(tspan,z0,sample, k1, k2)

	#Next with GPU
	z01G = np.array([2])
	z02G = np.array([3])
	tiG = np.array([0])
	tfG = np.array([10])
	k1G = np.array([0])
	k2G = np.array([1])  
	sampleG = np.array([.001])
	nG = np.array([int(math.ceil((tfG[0]-tiG[0])/sampleG[0])+1)]) 
	#m = len(z0)
	mG = np.array([2])
	iG = np.array([0])

	zzG = np.zeros((1,nG[0],mG[0]))
	ttG = np.zeros((1,nG[0],1))

	#Here we start to pass data from host(CPU) to device(GPU)
	d_z01G = cuda.to_device(z01G)
	d_z02G = cuda.to_device(z02G)
	d_tiG = cuda.to_device(tiG)
	d_tfG = cuda.to_device(tfG)
	d_k1G = cuda.to_device(k1G)
	d_k2G = cuda.to_device(k2G)
	d_sampleG = cuda.to_device(sampleG)
	d_nG = cuda.to_device(nG)
	d_mG = cuda.to_device(mG)
	d_iG = cuda.to_device(iG)

	d_zzG = cuda.to_device(zzG)
	d_ttG = cuda.to_device(ttG)

	#Even when execute only one Operation we define 32 simulations to send data needed by GPU
	blocks_per_grid = 16
	threads_per_block = 2
	odeFelGPU[blocks_per_grid, threads_per_block](d_z01G, d_z02G, d_tiG, 
		d_tfG, d_k1G, d_k2G, d_sampleG, d_nG, d_mG, d_iG, d_zzG, d_ttG)

	#transfer back to CPU to plot
	zzGPU = d_zzG[0,:,:].copy_to_host()
	ttGPU = d_ttG[0,:,:].copy_to_host()
	#pdb.set_trace()	
	plot(zz,tt,zzGPU, ttGPU )


@cuda.jit
def odeFelGPU(d_z01G, d_z02G, d_tiG, d_tfG, d_k1G, d_k2G, d_sampleG, d_nG, d_mG, d_iG, d_zzG, d_ttG):
	idx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x

	#Only perform calculations with index 0
	if idx==0:
		i = 0
		t = d_tiG[idx]
		d_ttG[idx,i] = t
		d_zzG[idx,i,0] = d_z01G[idx]
		d_zzG[idx,i,1] = d_z02G[idx]
		zp_ant = cuda.local.array((2),dtype=types.float32)
		fUpdateGPU(t,d_z01G,d_z02G, d_k1G, d_k2G, zp_ant)
		while(t<(d_tfG[idx]-d_sampleG[idx])):
			z0_ = cuda.local.array((1),dtype=types.float32)
			z1_ = cuda.local.array((1),dtype=types.float32)
			z0_[0]=d_zzG[idx,i,0]
			z1_[0]=d_zzG[idx,i,1]
			zp = cuda.local.array((2),dtype=types.float32)
			fUpdateGPU(t,z0_,z1_, d_k1G, d_k2G, zp)
			i+=1
			d_zzG[idx,i,0]= d_zzG[idx,i-1,0]+(zp[0]+zp_ant[0])*(d_sampleG[idx]/2)
			d_zzG[idx,i,1]= d_zzG[idx,i-1,1]+(zp[1]+zp_ant[1])*(d_sampleG[idx]/2)
			zp_ant = zp
			t+=d_sampleG[idx]
			d_ttG[idx,i]=t

@cuda.jit(device=True)
def fUpdateGPU(t,d_z01G,d_z02G, d_k1G, d_k2G, zp):
	z1p = -d_k1G[0]*d_z01G[0]-d_k2G[0]*d_z02G[0]
	z2p = d_k2G[0]*d_z01G[0]-d_k1G[0]*d_z02G[0]
	zp[0] = z1p
	zp[1] = z2p


def odeFelCPU(tspan,z0, sample, k1, k2):
	ti = tspan[0]
	tf = tspan[1]
	n= int(math.ceil((tf-ti)/sample)+1)
	m = len(z0)
	zz = np.zeros((n,m))
	tt = np.zeros((n,1))

	i= 0
	t=ti
	tt[i]=t
	zz[i,:] = z0

	zp_ant = fUpdateCPU(t,z0,k1,k2)
	#pdb.set_trace()
	while (t<(tf-sample)):
		z=zz[i,:]
		zp=fUpdateCPU(t,z,k1,k2)
		pdb.set_trace()
		i+=1
		zz[i,0] = zz[i-1,0]+(zp[0]+zp_ant[0])*(sample/2)
		zz[i,1] = zz[i-1,1]+(zp[1]+zp_ant[1])*(sample/2)
		#pdb.set_trace()
		zp_ant = zp
		t = t+sample 
		tt[i] = t
		#pdb.set_trace()

	#pdb.set_trace()
	return tt, zz

def fUpdateCPU(t,z,k1,k2):
	z1 = z[0]
	z2 = z[1]

	z1p = -k1*z1-k2*z2
	z2p = k2*z1-k1*z2

	zp=np.array([z1p, z2p])

	#pdb.set_trace()
	return zp

def plot(zz,tt, zzGPU, ttGPU):

	plt.figure("Integration", dpi=150)

	plt.subplot(2,3,1)
	plt.xlabel('Time')
	plt.ylabel("z1")
	plt.plot(tt,zz[:,0])

	plt.subplot(2,3,2)
	plt.xlabel('Time')
	plt.ylabel("z2")
	plt.plot(tt,zz[:,1])

	plt.subplot(2,3,3)
	plt.xlabel('z1')
	plt.ylabel("z2")
	plt.plot(zz[:,0],zz[:,1])

	plt.subplot(2,3,4)
	plt.xlabel('Time')
	plt.ylabel("z1")
	plt.plot(ttGPU,zzGPU[:,0])

	plt.subplot(2,3,5)
	plt.xlabel('Time')
	plt.ylabel("z2")
	plt.plot(ttGPU,zzGPU[:,1])

	plt.subplot(2,3,6)
	plt.xlabel('z1')
	plt.ylabel("z2")
	plt.plot(zzGPU[:,0],zzGPU[:,1])

	plt.show()





	




def testIntegration():
	# number of independent oscillators
	# to simulate
	trials = 10_000

	# time variables
	t0 = 0
	t_end = 100
	dt = 0.01
	t = np.array([t0, t_end, dt], dtype='float32')

	# generate random initial condiotions
	init_states = np.random.random_sample(2 * trials).astype('float32')

	# manage nr of threads (threads)
	threads_per_block = 32
	blocks_per_grid = \
	    (init_states.size + (threads_per_block - 1)) // threads_per_block

	# start timer
	#start = time.perf_counter()

	#pdb.set_trace()	

	# start parallel simulations
	solve_ode[blocks_per_grid, threads_per_block](init_states, t)

	# measure time elapsed
	#end = time.perf_counter()
	#print(f'The result was computed in {end-start} s')

	# reshape the array into 2D
	x = init_states.reshape((trials, 2))

	# plot the phase space
	plt.scatter(x[:, 0], x[:, 1], s=1)

@cuda.jit
def solve_ode(x, time):
    """
    Solve 2DoF ode on gpu, given
    the initial conditions. The
    result will be stored in the
    input array.
    

    Parameters
    ----------
    x : np.array
        Contains initial conditions for the simulations.
        The elements are arranged in pairs:
        [x1_sim1, x2_sim1, x1_sim2, x2_sim2, ...]
    time : np.array
        Three-element list with time details.

    Returns
    -------
    None.

    """
    # time variables
    t = time[0]
    t_end = time[1]
    dt = time[2]
    
    # index of thread on GPU
    pos = cuda.grid(1)
    # mappping index to access every
    # second element of the array
    pos = pos * 2
    
    # condidion to avoid threads
    # accessing indices out of array
    if pos < x.size:
        # execute until the time reaches t_end
        while t < t_end:
            # compute derivatives
            dxdt0 = x[pos+1]
            dxdt1 = np.float32(10.0)*math.sin(t) - np.float32(0.1)*x[pos+1] - x[pos]**3
            
            # update state vecotr
            x[pos] += dxdt0 * dt
            x[pos+1] += dxdt1 * dt
            
            # update time
            t += dt
            


def GPU_testGeneric(request):
	testGeneric()
	return render(request,'showmap.html')

def testGeneric():
	
	zz = np.zeros((121,90100,9), dtype=float)

	z0 = np.empty((90100,9), dtype=float)
	z0[0,0] = 1
	z0[0,1] = 2
	z0[0,2] = 3
	z0[0,3] = 4
	z0[0,4] = 5
	z0[0,5] = 6
	z0[0,6] = 7
	z0[0,7] = 8
	z0[0,8] = 9

	threads_per_block = 16
	blocks_per_grid = 16

	d_zz = cuda.to_device(zz)
	d_z0 = cuda.to_device(z0)

	testGeneric_GPU[blocks_per_grid, threads_per_block](d_z0, d_zz)

	zz = d_zz.copy_to_host()
	z0 = d_z0.copy_to_host()

	pdb.set_trace()


@cuda.jit
def testGeneric_GPU(d_z0, d_zz):
	idx = cuda.grid(1)

	if (idx == 0):
		d_zz[0,0] = d_z0[0]

