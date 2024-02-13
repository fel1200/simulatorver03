#GPU_integrateRespa2.py

#Functionality from time library
import time

#To debug
import pdb

#Numpy library
import numpy as np

#For map functionality
import folium
from django.shortcuts import render,redirect
from . import a_getroute

#To integrate

from scipy.integrate import odeint

#To use math functions
import math

#To plot values
import matplotlib.pyplot as plt 

#To use GPU paralelization capabilites
from numba import cuda, types

#To use os and activate GPU debug
import os


def GPU_integrate(request):
	#To measure time	
	start = time.perf_counter()
	initModelIntegrateGPU()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')

	return render(request,'showmap.html')


def initModelIntegrateGPU():
	#Setting environment var to enable debug in GPU
	os.environ['NUMBA_ENABLE_CUDASIM'] = "1"
	#pdb.set_trace()					

	#Globar vars
	global ALPHA_MAX
	ALPHA_MAX =.04
	global ALPHA_MIN
	ALPHA_MIN = .01
	
	#Instead this value, we use the value of rho by each segment
	global RHO
	RHO = .02

	global KI
	KI = 15

	global CHI 
	CHI = 0.7

	#static traffic light times
	global YELLOW_TIME
	YELLOW_TIME = 5
	global ALL_RED_TIME
	ALL_RED_TIME = 5
	global STEP_TIME
	STEP_TIME = .1

	global SUB_STEP_TIME
	SUB_STEP_TIME = .001

	#times of vertical and horizontal		
	global MIN_GREEN_TIME_HORIZONTAL
	MIN_GREEN_TIME_HORIZONTAL = 10

	global MAX_GREEN_TIME_HORIZONTAL		
	MAX_GREEN_TIME_HORIZONTAL = 11

	global MIN_GREEN_TIME_VERTICAL		
	MIN_GREEN_TIME_VERTICAL = 10

	global MAX_GREEN_TIME_VERTICAL		
	MAX_GREEN_TIME_VERTICAL = 11

	global TOTAL_TIME
	TOTAL_TIME = 900

	#To plot or not
	isPlotting = False

	#Number of samples in zd
	samplesbySimulation = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME))
	
	#Maximum number of cars by segment
	global ZMAX
	ZMAX = np.array([35.0,35.0,35.0,35.0,10.0,200.0
		,200.0,200.0,200.0]);

	#array of rho
	global RHOA
	RHOA = ZMAX*RHO
		
	#number Of simulations
	simulations = (MAX_GREEN_TIME_HORIZONTAL-MIN_GREEN_TIME_HORIZONTAL)*(MAX_GREEN_TIME_VERTICAL-MIN_GREEN_TIME_VERTICAL)
	#pdb.set_trace()

	counterSimulations = 0

	#if values are plotted
	weight52s = np.zeros((simulations,samplesbySimulation), dtype=float)
	#pdb.set_trace()					
	weight54s = np.zeros((simulations,samplesbySimulation), dtype=float)
	h15s = np.zeros((simulations,samplesbySimulation), dtype=float)
	h35s = np.zeros((simulations,samplesbySimulation), dtype=float)
	#index x = simulations derived by trafficlights, i.e. 1,681
	#index y = each segment i.e. 0..9
	#index z= samplesExecutedeachSimulation i.e. 1 / .01(subsub) = 100
	alphas = np.zeros((simulations,9,samplesbySimulation), dtype=float)
	aks = np.zeros((simulations,9,samplesbySimulation), dtype=float)

	#To save h15, h35, w52, w54, alpha and ak
	
	h15 = np.zeros((simulations), dtype=float)
	h35 = np.zeros((simulations), dtype=float)
	weight52 = np.zeros((simulations), dtype=float)
	weight54 = np.zeros((simulations), dtype=float)
	alpha = np.zeros((simulations,9), dtype=float)
	ak = np.zeros((simulations,9), dtype=float)
	alpha_diag = np.zeros((simulations, 9,9), dtype=float)
	ak_diag = np.zeros((simulations, 9,9), dtype=float)

	#Setting gammas in 0 for connection between segments
	G = np.zeros((simulations,9,9), dtype=float)

	buffer1 = np.zeros((simulations,9,9),dtype=float)
	buffer2 = np.zeros((simulations,9,9),dtype=float)
	buffer3 = np.zeros((simulations,9),dtype=float)

	#Empty array to save zp
	zps = np.empty((simulations,2,9), dtype=float)

	#Initial and current cars
	z0 = np.empty((samplesbySimulation,9), dtype=float)
	zd = np.empty((samplesbySimulation,9), dtype=float)


	zd_diag = np.empty((simulations,9,9), dtype=float)

	zpAnt = np.empty((simulations,9), dtype=float)

	#empty array with 0s to save zz
	zz = np.zeros((simulations,samplesbySimulation,9), dtype=float)
	tt = np.zeros((simulations,samplesbySimulation,1), dtype=float)
	#pdb.set_trace()	

	#to save traffic lights' parameters
	M_green_time_horizontal = np.empty((simulations), dtype=np.int16)
	M_green_time_vertical = np.empty((simulations), dtype=np.int16)
	#Initializating yellow and red times with 5 seconds
	M_yellow_time = np.full((simulations),5, dtype=np.int16)
	M_allred_time = np.full((simulations),5, dtype=np.int16)
	#pdb.set_trace()	

	i = 0

	for green_time_horizontal in range(MIN_GREEN_TIME_HORIZONTAL,MAX_GREEN_TIME_HORIZONTAL):
		for green_time_vertical in range(MIN_GREEN_TIME_VERTICAL,MAX_GREEN_TIME_VERTICAL):
			M_green_time_horizontal[i] = green_time_horizontal
			M_green_time_vertical[i] = green_time_vertical
			i +=1

		#Matrix definition
		Matrix2DToFillOutput = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1],	
					[0, 1, 0, 1, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0]])
		#Transpose to obtain the output of segments
		Matrix2DToFillInput = Matrix2DToFillOutput.T	

		#Values of in and out Matrix in each simulation
		#This changes from the previous CPU approach
		#Where only one var in each simulation was enough
		MAout = np.empty((simulations,9,9), dtype=float)
		MAin = np.empty((simulations,9,9), dtype=float)
	
		#Cycle to fill zCars, z0, MAout
		for simulation in range(0,simulations):
			zd[simulation,0] = z0[simulation,0]= 0.0
			zd[simulation,1] = z0[simulation,1]=0.0
			zd[simulation,2] = z0[simulation,2]=0.0
			zd[simulation,3] = z0[simulation,3]=0.0
			zd[simulation,4] = z0[simulation,4]=0.0
			zd[simulation,5] = z0[simulation,5]=100.0
			zd[simulation,6] = z0[simulation,6]=100.0
			zd[simulation,7] = z0[simulation,7]=0.0
			zd[simulation,8] = z0[simulation,8]=0.0
			#Matrix fill 
			MAout[simulation:,] = Matrix2DToFillOutput
			MAin[simulation:,] = Matrix2DToFillInput
	

		#To save temporary results of matrix operations 
		POSM = np.empty((simulations,9,9), dtype = np.float32)
		NEGM = np.empty((simulations,9,9), dtype = np.float32)

		#To save temporary results of matrix operations 
		POS = np.empty((simulations,9), dtype = np.float32)
		NEG = np.empty((simulations,9), dtype = np.float32)

		zp = np.empty((simulations,9), dtype = np.float32)
		zp_ant = np.empty((simulations,9), dtype = np.float32)

		#pdb.set_trace()	

		#Here it begins the best
		#where children becomes men
		#and men become heroes :)
		#and heroes start crying
		#i.e. GPU parameters
		blocks_per_grid = 32
		threads_per_block = int(math.ceil(simulations/blocks_per_grid))
		#pdb.set_trace()	

		#Moving data from host to device
		d_MAout = cuda.to_device(MAout)
		d_MAin = cuda.to_device(MAin)

		d_M_green_time_horizontal = cuda.to_device(M_green_time_horizontal)
		d_M_green_time_vertical = cuda.to_device(M_green_time_vertical)

		d_M_yellow_time = cuda.to_device(M_yellow_time)
		d_M_allred_time = cuda.to_device(M_allred_time)

		d_h15 = cuda.to_device(h15)
		d_h35 = cuda.to_device(h35)
		d_weight52 = cuda.to_device(weight52)
		d_weight54 = cuda.to_device(weight54)

		d_alpha = cuda.to_device(alpha)
		d_ak = cuda.to_device(ak)
		d_alpha_diag = cuda.to_device(alpha_diag)
		d_ak_diag = cuda.to_device(ak_diag)


		d_G = cuda.to_device(G)

		d_buffer1 = cuda.to_device(buffer1)
		d_buffer2 = cuda.to_device(buffer2)
		d_buffer3 = cuda.to_device(buffer3)

		d_zps = cuda.to_device(zps)

		d_z0 = cuda.to_device(z0)
		d_zpAnt = cuda.to_device(zpAnt)

		d_zz = cuda.to_device(zz)
		d_tt = cuda.to_device(tt)

		#Scalars passed as one value array
		a_ALPHA_MAX = np.array([ALPHA_MAX])
		a_ALPHA_MIN = np.array([ALPHA_MIN])
		
		#a_RHO = np.array([RHO])
		a_KI = np.array([KI])
		a_CHI = np.array([CHI])

		d_a_ALPHA_MAX = cuda.to_device(a_ALPHA_MAX)
		d_a_ALPHA_MIN = cuda.to_device(a_ALPHA_MIN)

		d_RHO = cuda.to_device(RHOA)
		d_a_KI = cuda.to_device(a_KI)
		d_a_CHI = cuda.to_device(a_CHI)

		d_ZMAX = cuda.to_device(ZMAX)

		t= 0.00
		#Time var saved in an array
		a_time = np.zeros((simulations,1), dtype=np.float16)
		d_time = cuda.to_device(a_time)

		d_POSM = cuda.to_device(POSM)
		d_NEGM = cuda.to_device(NEGM)

		d_POS = cuda.to_device(POS)
		d_NEG = cuda.to_device(NEG)

		d_zp = cuda.to_device(zp)
		d_zp_ant = cuda.to_device(zp_ant)

		#Number of samples in zd
		samples = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME)	)
	
		zpSimulations = np.empty((samples,12),dtype=float)

		#while(t<TOTAL_TIME):

		current_time = np.array([0.00])
		d_current_time = cuda.to_device(current_time)

		i = np.array([0])

		#while(t<(t+SUB_STEP_TIME)):
		while(current_time[0] < 1700):	
			
			limit_time = np.array([current_time[0]+30])
			#Updating time eachCycle
			d_time = cuda.to_device(a_time)
			d_limit_time = cuda.to_device(limit_time)			

			d_i = cuda.to_device(i)

			#pdb.set_trace()	
			#Updating zd to parallel processing
			d_zd = cuda.to_device(zd)
			d_zd_diag = cuda.to_device(zd_diag)

			tspan =np.linspace(t,TOTAL_TIME,int(TOTAL_TIME/SUB_STEP_TIME))
			#tspan =np.linspace(t,t+STEP_TIME,int(STEP_TIME/SUB_STEP_TIME))
			#pdb.set_trace()	
			d_tspan = cuda.to_device(tspan)			
			
			
			stepGPUIntegrate[blocks_per_grid, threads_per_block](d_time, 
				d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
				d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
				 d_M_green_time_horizontal, 
				 d_M_green_time_vertical,
				 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52, d_weight54,
				 d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, d_POSM, d_NEGM, d_tspan, d_alpha_diag, d_ak_diag, d_zp_ant, d_zp, d_zz, d_tt, d_current_time, d_limit_time, d_i)
			#transfer back to CPU to check info
			current_time = d_current_time.copy_to_host()
			limit_time = d_limit_time.copy_to_host()


			POSM = d_POSM.copy_to_host()
			NEGM = d_NEGM.copy_to_host()
			POS = d_POS.copy_to_host()
			NEG = d_NEG.copy_to_host()

			
			zp_ant = d_zp_ant.copy_to_host()
			zp = d_zp.copy_to_host()

			i = d_i.copy_to_host()


			a_time = d_time.copy_to_host()

			zz = d_zz.copy_to_host()
			tt = d_tt.copy_to_host()

			z0 = zz[0,0:i[0],]
			
			zd = d_zd.copy_to_host()
			zd_diag = d_zd_diag.copy_to_host()

			h15 = d_h15.copy_to_host()
			h35 = d_h35.copy_to_host()
			weight52 = d_weight52.copy_to_host()
			weight54 = d_weight54.copy_to_host()

			alpha = d_alpha.copy_to_host()
			ak = d_ak.copy_to_host()
			alpha_diag = d_alpha_diag.copy_to_host()
			ak_diag = d_ak_diag.copy_to_host()

			G = d_G.copy_to_host()

			buffer1 = d_buffer1.copy_to_host()
			buffer2 = d_buffer2.copy_to_host()
			buffer3 = d_buffer3.copy_to_host()

			MAout = d_MAout.copy_to_host()
			MAin = d_MAin.copy_to_host()
			
			#pdb.set_trace()	
			#if (isPlotting):
			#	plot(zz[0,0:3000,], h15s, h35s, weight54s, weight52s)

			'''for simulation in range(simulations):
				zps[simulation,] =a odeint(updateCycle, zd[simulation], tspan,
				args=())
				zp = zps[simulation]
				zd[simulation] = zp[-1]'''
			#print(t)	
			t +=SUB_STEP_TIME
		np.savetxt("zz.csv", zz[0,0:170000],delimiter = ',')
		if (isPlotting):
			plot(zz[0,0:170000,], h15s, h35s, weight54s, weight52s)
			#plot(zps, h15s, h35s, weight54s, weight52s)
			#plot(zps, h15s, h35s, weight54s, weight52s)

@cuda.jit(device=True)
def getState(d_time,d_M_green_time_horizontal, d_M_yellow_time, d_M_allred_time,d_M_green_time_vertical, d_ak, idx ):
	weight52Device = 0.00
	weight54Device = 0.00
	h_15Device = 0.00
	h_35Device = 0.00
	div = int(d_time[idx,0]/(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+
			d_M_yellow_time[idx]+d_M_allred_time[idx]))
	timeSM = d_time[idx,0]-(div * (d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+
			d_M_yellow_time[idx]+d_M_allred_time[idx]))
	if (timeSM>0 and timeSM<(d_M_green_time_horizontal[idx]+
		d_M_yellow_time[idx])):
		#currentState = 1
		weight52Device = 0.6
		weight54Device = 0.4
		h_15Device = 1
		h_35Device = 0
	elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]) and 
		timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx])):
		#currentState = 2
		weight52Device = 0.6
		weight54Device = 0.4
		h_15Device = 0
		h_35Device = 0
	elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]) 
		and (timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]))):
		#currentState = 3
		weight52Device = 0.4
		weight54Device = 0.6
		h_15Device = 0
		h_35Device = 1
	elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]) 
		and (timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]))):
		#currentState = 4
		weight52Device = 0.4
		weight54Device = 0.6
		h_15Device = 0
		h_35Device = 0
	else:
		#currentState = 1
		#time=0
		weight52Device = 0.6
		weight54Device = 0.4
		h_15Device = 1
		h_35Device = 0
	return weight52Device, weight54Device, h_15Device, h_35Device

@cuda.jit(device=True)
def getAlpha_Ak(d_z0, d_zd, d_ZMAX, d_alpha, d_RHO, d_a_ALPHA_MAX, d_a_ALPHA_MIN, d_ak, idx):
	for i in range(0,9):
		#checking if z isn't greater than MAX values
		if d_z0[idx,i]>d_ZMAX[i]:
			d_z0[idx,i]=ZMAX[i]
		elif d_z0[idx,i] < 0.00:
			d_z0[idx,i] = 0.00

		#Copying z0 to zd as a backup array to make calculations further
		d_zd[idx,i] = d_z0[idx,i]

		#Getting alpha values
		d_alpha[idx,i] = 0.00
		if (d_zd[idx,i] < d_RHO[i]):
			d_alpha[idx,i] = d_a_ALPHA_MAX[0]
		elif ((d_zd[idx,i] > d_RHO[i]) and 
			(d_zd[idx,i] <= d_ZMAX[i])):
			d_alpha[idx,i] = ((d_a_ALPHA_MAX[0]-d_a_ALPHA_MIN[0])/(d_RHO[i]-d_ZMAX[i]))*(d_zd[idx,i]-d_ZMAX[i])+d_a_ALPHA_MIN[0]
		elif ((d_zd[idx,i]>d_ZMAX[i])):
			d_alpha[idx,i] = d_a_ALPHA_MIN[0]

		#Getting ak values	
		ak = 1.0
		zk = d_zd[idx,i]
		delta = (15*((zk/d_ZMAX[i]) - 0.7))
		ei =math.exp(delta)
		d_ak[idx,i] = 1/(1+ei)

@cuda.jit(device=True)
def matrixOperations(d_alpha, d_ak, d_MAin, d_MAout, d_G, d_buffer1, d_buffer2, d_buffer3, d_zd, d_zd_diag, d_POS, d_NEG, d_alpha_diag, d_ak_diag, idx):
	
	#cleaning up buffers
	d_buffer1[:] = 0
	d_buffer2[:] = 0
	d_buffer3[:] = 0

	#cleaning up POS and NEG
	d_POS[:] = 0
	d_NEG[:]=0

	#Diagonal operations
	#d_alpha_diag = cuda.local.array((9,9),dtype=types.float32)
	#d_ak_diag = cuda.local.array((9,9),dtype=types.float32)
	device_diag(d_alpha[idx,], d_alpha_diag, idx)
	device_diag(d_ak[idx,], d_ak_diag, idx)

	
	#to make operations wi th an specific matrix
	#d_buffer1 = cuda.local.array((9,9),dtype=types.float32)
	#d_buffer2 = cuda.local.array((9,9),dtype=types.float32)
	#d_buffer3 = cuda.local.array((9),dtype=types.float32)
	
	#np.multiply = element-wise mutiplication
	#np.dot = usual matrix multiplication

	#buffer1 save np.multiply(d_MAin[idx,],G[idx,])
	element_matrix_mult(idx,d_MAin,d_G, d_buffer1)
	
	
	#buffer2 save (Mak.dot(np.multiply(d_MAin[idx,],G[idx,])))
	usual_matrix_mult(d_ak_diag[idx,] ,d_buffer1[idx,], d_buffer2, idx)
	
	#buffer3 save Malpha.dot(d_zd[idx])
	usual_matrix_mult1d_idx(d_alpha_diag[idx,],d_zd[idx,], d_buffer3, idx)
	
	
	#buffer4 save (Mak.dot(np.multiply(d_MAin[idx,],G[idx,]))).dot(Malpha.dot(d_zd[idx]))
	#This sentence complete first operations:
	#POS = (Mak.dot(np.multiply(d_MAin[idx,],G[idx,]))).dot(Malpha.dot(d_zd[idx]))		
	usual_matrix_mult1d_idx(d_buffer2[idx,],d_buffer3[idx,], d_POS, idx)
	
	#Here we go with NEGD
	#np.diag(d_zd[idx]
	#zd diag
	#d_zd_diag = cuda.local.array((9,9),dtype=types.float32)
	device_diag(d_zd[idx,], d_zd_diag, idx)
	
	#cleaning up buffers
	d_buffer1[:] = 0
	d_buffer2[:] = 0
	d_buffer3[:] = 0


	#-Malpha.dot(np.diag(d_zd[idx])		
	usual_matrix_mult(d_alpha_diag[idx,], d_zd_diag[idx,] , d_buffer1, idx)		
	
	#np.multiply(d_MAout[idx,],G[idx,])
	element_matrix_mult(idx,d_MAout,d_G, d_buffer2)
	
	


	#buffer3 save (np.multiply(d_MAout[idx,],G[idx,]).dot(np.diag(Mak)))
	#operation with Ak like vector no matrix, because
	#in original operation is a double diag

	usual_matrix_mult1d_idx(d_buffer2[idx,],d_ak[idx,], d_buffer3, idx)
	
	#Final operation to get NEG
	#NEG = -Malpha.dot(np.diag(d_zd[idx]).dot(np.multiply(d_MAout[idx,],G[idx,]).dot(np.diag(Mak))))
	usual_matrix_mult1d_idx(d_buffer1[idx,],d_buffer3[idx,], d_NEG, idx)
	"""
	"""

@cuda.jit(device=True)
def defineGammas(d_G, d_h15, d_h35, d_weight52, d_weight54, idx):
		d_G[idx,0,4] = d_h15[idx] #Gamma for connection 1,5
		d_G[idx,4,0] = d_h15[idx] #Gamma for connection 1,5
		d_G[idx,2,4] = d_h35[idx] #Gamma for connection 3,5
		d_G[idx,4,2] = d_h35[idx] #Gamma for connection 3,5
		d_G[idx,4,1] = d_weight52[idx] #Gamma weight for connection 5,2
		d_G[idx,1,4] = d_weight52[idx] #Gamma weight for connection 5,2
		d_G[idx,4,3] = d_weight54[idx] #Gamma for connection 5,4 
		d_G[idx,3,4] = d_weight54[idx] #Gamma for connection 5,4

		d_G[idx,1,7]= 1 #Gamma for connection 2,8
		d_G[idx,7,1]= 1 #Gamma for connection 2,8
		d_G[idx,3,8]= 1 #Gamma for connection 4,9
		d_G[idx,8,3]= 1 #Gamma for connection 4,9
		d_G[idx,5,0]= 1 #Gamma for connection 6,1
		d_G[idx,0,5]= 1 #Gamma for connection 6,1
		d_G[idx,6,2]= 1 #Gamma for connection 7,3
		d_G[idx,2,6]= 1 #Gamma for connection 7,3

@cuda.jit(device=True)
def funczp(d_time, 
			d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
			d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
			 d_M_green_time_horizontal, 
			 d_M_green_time_vertical,
			 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52,
			 d_weight54,d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, 
			 d_POSM, d_NEGM, d_tspan, d_alpha_diag, d_ak_diag, idx):
	#First status updating
	d_weight52[idx],d_weight54[idx], d_h15[idx], d_h35[idx] = getState(d_time,d_M_green_time_horizontal, d_M_yellow_time, d_M_allred_time,d_M_green_time_vertical, d_ak, idx)

	#Then update alpha and ak values
	getAlpha_Ak(d_z0, d_zd, d_ZMAX, d_alpha, d_RHO, d_a_ALPHA_MAX, d_a_ALPHA_MIN, d_ak, idx)

	#Defining gammas for intersection connections
	defineGammas(d_G, d_h15, d_h35, d_weight52, d_weight54, idx)

	matrixOperations(d_alpha, d_ak, d_MAin, d_MAout, d_G, d_buffer1, d_buffer2, d_buffer3, d_zd, d_zd_diag, d_POS, d_NEG, d_alpha_diag, d_ak_diag,idx)

from pdb import set_trace

#Kernel GPU
@cuda.jit(debug=True)
def stepGPUIntegrate(d_time, 
				d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
				d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
				 d_M_green_time_horizontal, 
				 d_M_green_time_vertical,
				 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52, d_weight54,
				 d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag,
				  d_z0, d_POS, d_NEG, d_POSM, d_NEGM, d_tspan, d_alpha_diag, d_ak_diag, d_zp_ant, d_zp, d_zz, d_tt, d_current_time, d_limit_time,d_i):
	#getting thread index
	idx = cuda.blockIdx.x*cuda.blockDim.x +cuda.threadIdx.x
	
	#idx lower than simulations
	#if idx< d_M_green_time_horizontal.shape[0]:
	if idx ==0:
		d_time[idx,0] = d_current_time[0]
		d_z = cuda.local.array((9),dtype=types.float32)
		#di = cuda.local.array((121),dtype=types.int32)

		for j in range(0,9):
			if d_z0[idx,j]>d_ZMAX[j]:
				#pdb.set_trace()		
				d_z0[idx,j]=d_ZMAX[j]
			elif d_z0[idx,j] < 0.00:
				d_z0[idx,j] = 0.00


		funczp(d_time, 
			d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
			d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
			 d_M_green_time_horizontal, 
			 d_M_green_time_vertical,
			 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52,
			 d_weight54,d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, 
			 d_POSM, d_NEGM, d_tspan, d_alpha_diag, d_ak_diag, idx)
		
		for j in range(9):
			d_zz[idx,0,j] = d_z0[idx,j]
			d_zp_ant[idx,j] = d_POS[idx,j]-d_NEG[idx,j]
		#d_i[0] = 0
		#i = 0	
		
		#while d_time[idx,0] < (30): #For testing purposes
		while d_current_time[0] < d_limit_time[0]:
			for j in range(9):
				d_z0[idx,j] = d_zz[idx,d_i[0],j]
				d_zd[idx,j] = d_zz[idx,d_i[0],j]
				if d_z0[idx,j]>d_ZMAX[j]:
					#pdb.set_trace()		
					d_z0[idx,j]=d_ZMAX[j]
				elif d_z0[idx,j] < 0.00:
					d_z0[idx,j] = 0.00

			funczp(d_time, 
				d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
				d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
				 d_M_green_time_horizontal, 
				 d_M_green_time_vertical,
				 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52,
				 d_weight54,d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, 
				 d_POSM, d_NEGM, d_tspan, d_alpha_diag, d_ak_diag, idx)
			d_i[0] = d_i[0]+1
			for j in range(9):
				d_zp[idx,j] = d_POS[idx,j]-d_NEG[idx,j]
				d_zz[idx,d_i[0],j] = d_zz[idx,d_i[0]-1,j]+(d_zp[idx,j]+d_zp_ant[idx,j])*(.01 /2)
				d_zp_ant[idx,j] =d_zp[idx,j]
			d_tt[idx,d_i[0]] = d_time[idx,0] 
			#d_time[idx,0] = d_time[idx,0] + .01
			d_current_time[0] = d_current_time[0] +.01




#To be called only from device
@cuda.jit(device=True)
def device_diag(original,diagonal, idx):
	for x in range(original.shape[0]):
		for y in range(original.shape[0]):
			if x == y:
				diagonal[idx,x,y] = original[x]			

#To be called only from device
#
@cuda.jit(device=True)
def element_matrix_mult(idx,matrix1,matrix2, result):
	for x in range(matrix1.shape[1]):
		for y in range(matrix1.shape[2]):
			result[idx,x,y] = matrix1[idx,x,y]*matrix2[idx,x,y]

@cuda.jit(device=True)
def element_matrix_mult_buffer(matrix1,matrix2,result):
	for x in range(matrix1.shape[0]):
		for y in range(matrix1.shape[1]):
			result[x,y] = matrix1[x,y]*matrix2[x,y]

	
@cuda.jit(device=True)
def usual_matrix_mult(matrix1,matrix2,result, idx):
	for x in range(matrix1.shape[0]):
		for y in range(matrix2.shape[1]):
			for z in range(matrix2.shape[0]):
				result[idx,x,y] += matrix1[x,z]*matrix2[z,y]


@cuda.jit(device=True)
def usual_matrix_mult1d(matrix1,matrix2, result):
	#sum = 0.00
	for x in range(matrix1.shape[0]):
		for y in range(matrix2.shape[0]):
			result[x] += matrix1[x,y]*matrix2[y]

@cuda.jit(device=True)
def usual_matrix_mult1d_idx(matrix1,matrix2, result, idx):
	#sum = 0.00
	for x in range(matrix1.shape[0]):
		for y in range(matrix2.shape[0]):
			result[idx,x] += matrix1[x,y]*matrix2[y]


@cuda.jit(device=True)
def usual_matrix_mult_buffer(idx,matrix1,matrix2, result):
	#sum = 0.00
	for x in range(matrix1.shape[1]):
		for y in range(matrix2.shape[2]):
			for z in range(matrix.shape[2]):
				result[x,y] += matrix1[x,z]*matrix2[z,y]

def updateCycle(z0,timeS):
	#pdb.set_trace()
	return z0





@cuda.jit
def integrateGPU():
	pass



def plot(zps, h15s, h35s, weight54s, weight52s):
		
		
				
		plt.figure("Zp", dpi=150)

		plt.subplot(3,4,1)
		plt.xlabel('Time')
		plt.ylabel("z7")
		plt.plot(zps[:,6])

		plt.subplot(3,4,2)
		plt.xlabel('Time')
		plt.ylabel("z3")
		plt.plot(zps[:,2])

		plt.subplot(3,4,3)
		plt.xlabel('Time')
		plt.ylabel("z4")
		plt.plot(zps[:,3])

		plt.subplot(3,4,4)
		plt.xlabel('Time')
		plt.ylabel("z9")
		plt.plot(zps[:,8])

		plt.subplot(3,4,5)
		plt.xlabel('Time')
		plt.ylabel("z6")
		plt.plot(zps[:,5])

		plt.subplot(3,4,6)
		plt.xlabel('Time')
		plt.ylabel("z1")
		plt.plot(zps[:,0])

		plt.subplot(3,4,7)
		plt.xlabel('Time')
		plt.ylabel("z2")
		plt.plot(zps[:,1])

		plt.subplot(3,4,8)
		plt.xlabel('Time')
		plt.ylabel("z8")
		plt.plot(zps[:,7])

		plt.subplot(3,4,9)
		plt.xlabel('Time')
		plt.ylabel("z5")
		plt.plot(zps[:,4])

		#plt.show()

		"""
		plt.figure("Flow horizontal-vertical weight")
		plt.subplot(4,1,1)
		plt.xlabel('Time')
		plt.ylabel("h15")
		plt.plot(h15s)

		plt.subplot(4,1,2)
		plt.xlabel('Time')
		plt.ylabel("h35")
		plt.plot(h35s)

		plt.subplot(4,1,3)
		plt.xlabel('Time')
		plt.ylabel("w52")
		plt.plot(weight52s)

		plt.subplot(4,1,4)
		plt.xlabel('Time')
		plt.ylabel("w54")
		plt.plot(weight54s)"""

		plt.show()







			funczp(d_time, 
				d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
				d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
				 d_M_green_time_horizontal, 
				 d_M_green_time_vertical,
				 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52,
				 d_weight54,d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3,
	             d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, 
				 d_tspan, d_alpha_diag, d_ak_diag, idx)
			
			for j in range(9):
				d_zz[idx,0,j] = d_z0[idx,j]
				d_zp_ant[idx,j] = d_POS[idx,j]-d_NEG[idx,j]
			#d_i[0] = 0
			#i = 0	
			
			#while d_time[idx,0] < (30): #For testing purposes
			while d_current_time[0] < d_limit_time[0]:
				for j in range(9):
					d_z0[idx,j] = d_zz[idx,d_i[0],j]
					d_zd[idx,j] = d_zz[idx,d_i[0],j]
					if d_z0[idx,j]>d_ZMAX[j]:
						#pdb.set_trace()		
						d_z0[idx,j]=d_ZMAX[j]
					elif d_z0[idx,j] < 0.00:
						d_z0[idx,j] = 0.00

				funczp(d_time, 
					d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
					d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
					 d_M_green_time_horizontal, 
					 d_M_green_time_vertical,
					 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52,
					 d_weight54,d_alpha,d_ak, d_G, d_buffer1, d_buffer2, d_buffer3, d_zps, d_zd, d_zd_diag, d_z0, d_POS, d_NEG, 
					 d_tspan, d_alpha_diag, d_ak_diag, idx)
				d_i[0] = d_i[0]+1
				for j in range(9):
					d_zp[idx,j] = d_POS[idx,j]-d_NEG[idx,j]
					d_zz[idx,d_i[0],j] = d_zz[idx,d_i[0]-1,j]+(d_zp[idx,j]+d_zp_ant[idx,j])*(.01 /2)
					d_zp_ant[idx,j] =d_zp[idx,j]
				d_tt[idx,d_i[0]] = d_time[idx,0] 
				#d_time[idx,0] = d_time[idx,0] + .01
				d_current_time[0] = d_current_time[0] +.01
