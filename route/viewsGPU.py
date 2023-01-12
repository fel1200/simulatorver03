#Functionality from time library
import time

#To debug
import pdb

#Numpy library
import numpy as np

#For map functionality
import folium
from django.shortcuts import render,redirect
from . import getroute

#To integrate

from scipy.integrate import odeint

#To use math functions
import math

#To plot values
import matplotlib.pyplot as plt 

#To use GPU paralelization capabilites
from numba import cuda, types



def pythonGPU(request):
	#To measure time	
	start = time.perf_counter()
	initModelArray()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')

	return render(request,'showmap.html')


def initModelArray():
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
	MAX_GREEN_TIME_HORIZONTAL = 21

	global MIN_GREEN_TIME_VERTICAL		
	MIN_GREEN_TIME_VERTICAL = 10

	global MAX_GREEN_TIME_VERTICAL		
	MAX_GREEN_TIME_VERTICAL = 21

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

	#Setting gammas in 0 for connection between segments
	G = np.zeros((simulations,9,9), dtype=float)


	#Empty array to save zp
	zps = np.empty((simulations,2,9), dtype=float)

	#Initial and current cars
	z0 = np.empty((simulations,9), dtype=float)
	zd = np.empty((simulations,9), dtype=float)

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

		#pdb.set_trace()	


		#Here it starts good!
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


		d_G = cuda.to_device(G)

		d_zps = cuda.to_device(zps)

		d_z0 = cuda.to_device(z0)

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
		a_time = np.array([t], dtype=np.float16)
		d_time = cuda.to_device(a_time)

		d_POSM = cuda.to_device(POSM)
		d_NEGM = cuda.to_device(NEGM)

		#Number of samples in zd
		samples = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME)	)
	
		zpSimulations = np.empty((samples,12),dtype=float)

		while(t<TOTAL_TIME):

			#Updating time eachCycle
			a_time[0] = t
			#Launching GPU numba kernel
			d_time = cuda.to_device(a_time)

			#pdb.set_trace()	
			#Updating zd to parallel processing
			d_zd = cuda.to_device(zd)
			stepGPU[blocks_per_grid, threads_per_block](d_time, 
				d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
				d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
				 d_M_green_time_horizontal, 
				 d_M_green_time_vertical,
				 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52, d_weight54,
				 d_alpha,d_ak, d_G, d_zps, d_zd, d_z0, d_POSM, d_NEGM)
			#To check results in host
			#weight52 = d_weight52.copy_to_host()
			
			#To check results
			"""
			zd = d_zd.copy_to_host()
			alpha = d_alpha.copy_to_host()
			ak = d_ak.copy_to_host()
			weight52 = d_weight52.copy_to_host()
			weight54 = d_weight54.copy_to_host()
			h15 = d_h15.copy_to_host()
			h35 = d_h35.copy_to_host()
			"""


			#alpha_diag = d_alpha_diag.copy_to_host()
			#ak_diag = d_ak_diag.copy_to_host()

			#pdb.set_trace()	

			#First integration only with one value
			
			tspan =np.linspace(t,t+SUB_STEP_TIME,2)

			

	#		for green_time_horizontal in range(MIN_GREEN_TIME_HORIZONTAL,MAX_GREEN_TIME_HORIZONTAL):
	#			for green_time_vertical in range(MIN_GREEN_TIME_VERTICAL,MAX_GREEN_TIME_VERTICAL):
					#totalTime of traffic light cycles
	#				totalTimeTrafficLights = green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME			

					#time counter in cycle
	#				t=0.00
	#				counter = 0

					#Matlab data to 1500 simulations, in python 900
					
				#tspan =np.linspace(t,t+STEP_TIME,int(STEP_TIME/SUB_STEP_TIME))
					
			for simulation in range(simulations):
				#pdb.set_trace()
				zps[simulation,] = odeint(updateCycle, zd[simulation], tspan,
				args=())
				#pdb.set_trace()
				zp = zps[simulation]
				zd[simulation] = zp[-1]

				#pdb.set_trace()	
				#zd[simulation] = zps[-1]
			#z0 is the last value of odeint
			

			#Saving data in zps	
			#zps = np.append(zps,zp, axis=0)

			#counter+=1

			#counterSimulations +=1
			#print(counterSimulations)

			#instead of SUB, we use subsub to integrate each value
			#in GPU simulation		
			t +=SUB_STEP_TIME
			#print(t)
			if (isPlotting):
				plot(zps, h15s, h35s, weight54s, weight52s)

#Trying to define a type to np array in kernel execution
dt = np.dtype('float32')


#Kernel numba decorator
#Main function
@cuda.jit
def stepGPU(d_time, 
			d_a_ALPHA_MAX,d_a_ALPHA_MIN, d_RHO, 
			d_a_KI, d_a_CHI, d_ZMAX, d_MAout,d_MAin,
			 d_M_green_time_horizontal, 
			 d_M_green_time_vertical,
			 d_M_yellow_time, d_M_allred_time,d_h15, d_h35,d_weight52, d_weight54,
			 d_alpha,d_ak, d_G, d_zps, d_zd, d_z0, d_POSM, d_NEGM):
	
	idx = cuda.blockIdx.x*cuda.blockDim.x +cuda.threadIdx.x

	#if idx == 0:
	#	from pdb import set_trace; set_trace()	
	#Checking index to access only data according with the number of simulation
	if idx< d_M_green_time_horizontal.shape[0]:
		#State machine function
		#In the future check if could be defined a separate function
		div = int(d_time[0]/(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+
			d_M_yellow_time[idx]+d_M_allred_time[idx]))
		timeSM = d_time[0]-(div * (d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
			d_M_allred_time[idx]+d_M_green_time_vertical[idx]+
			d_M_yellow_time[idx]+d_M_allred_time[idx]))


		if (timeSM>0 and timeSM<(d_M_green_time_horizontal[idx]+
			d_M_yellow_time[idx])):
			#currentState = 1
			d_weight52[idx] = 0.6
			d_weight54[idx] = 0.4
			d_h15[idx] = 1
			d_h35[idx] = 0
		elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]) and 
			timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx])):
			#currentState = 2
			d_weight52[idx] = 0.6
			d_weight54[idx] = 0.4
			d_h15[idx] = 0
			d_h35[idx] = 0
		elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx]) 
			and (timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]))):
			#currentState = 3
			d_weight52[idx] = 0.4
			d_weight54[idx] = 0.6
			d_h15[idx] = 0
			d_h35[idx] = 1
		elif (timeSM>=(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]) 
			and (timeSM<(d_M_green_time_horizontal[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx]+d_M_green_time_vertical[idx]+d_M_yellow_time[idx]+
				d_M_allred_time[idx]))):
			#currentState = 4
			d_weight52[idx] = 0.4
			d_weight54[idx] = 0.6
			d_h15[idx] = 0
			d_h35[idx] = 0
		else:
			#currentState = 1
			#time=0
			d_weight52[idx] = 0.6
			d_weight54[idx] = 0.4
			d_h15[idx] = 1
			d_h35[idx] = 0

		for i in range(0,9):
			#First checking if z isn't greater or lower that alloweda values
			if d_z0[idx,i]>d_ZMAX[i]:
				d_z0[idx,i]=ZMAX[i]
			elif d_z0[idx,i] < 0.00:
				d_z0[idx,i] = 0.00
			#Copying z0 to zd as a backup array to make calculations 
			#in next steps
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
			
		#Matrix diagonal of alpha and ak
		
		#alphaT = d_alpha[idx,].T
		#alphaT = np.empty((9), dt)


		#alphaT = np.array([[]], dtype=np.float64)
		#alphaT = np.zeros((9),dtype=np.float32)
		#alphaT = np.empty((9))
		#alphaT = d_alpha[idx,]
		
		#Malpha= np.diag(alphaT)
		#Malpha= np.diag(d_ak[idx,])
		
		#At CUDA some matrix functions are not available
		#Then we need to recreate diagonal function for alpha and ak
		#for x in range(9):
		#	for y in range(9):
		#		if x == y:
		#			d_alpha_diag[idx,x,y] = d_alpha[idx,x]
		#			d_ak_diag[idx,x,y] = d_alpha[idx,x]

		d_alpha_diag = cuda.local.array((9,9),dtype=types.float32)
		d_ak_diag = cuda.local.array((9,9),dtype=types.float32)

		device_diag(d_alpha[idx,], d_alpha_diag)
		device_diag(d_ak[idx,], d_ak_diag)
		
		#Defining gammas for intersection connections
		
		d_G[idx,0,4] = d_h15[idx] #Gamma for connection 1,5
		d_G[idx,4,0] = d_h15[idx] #Gamma for connection 1,5
		d_G[idx,2,4] = d_h35[idx] #Gamma for connection 3,5
		d_G[idx,4,2] = d_h35[idx] #Gamma for connection 3,5
		d_G[idx,4,1] = d_weight52[idx] #Gamma weight for connection 5,2
		d_G[idx,1,4] = d_weight52[idx] #Gamma weight for connection 5,2
		d_G[idx,4,3] = d_weight54[idx] #Gamma for connection 5,4 
		d_G[idx,3,4] = d_weight54[idx] #Gamma for connection 5,4
		
		#to make operations with an specific matrix
		d_buffer1 = cuda.local.array((9,9),dtype=types.float32)
		d_buffer2 = cuda.local.array((9,9),dtype=types.float32)
		
		d_buffer3 = cuda.local.array((9),dtype=types.float32)
		d_POS = cuda.local.array((9),dtype=types.float32)
		d_NEG = cuda.local.array((9),dtype=types.float32)
		
		#np.multiply = element-wise mutiplication
		#np.dot = usual matrix multiplication

		#buffer1 save np.multiply(d_MAin[idx,],G[idx,])
		element_matrix_mult(idx,d_MAin,d_G, d_buffer1)

		#buffer2 save (Mak.dot(np.multiply(d_MAin[idx,],G[idx,])))
		usual_matrix_mult(d_ak_diag ,d_buffer1, d_buffer2)

		#buffer3 save Malpha.dot(d_zd[idx])
		usual_matrix_mult1d(d_alpha_diag,d_zd[idx,], d_buffer3)

		#buffer4 save (Mak.dot(np.multiply(d_MAin[idx,],G[idx,]))).dot(Malpha.dot(d_zd[idx]))
		#This sentence complete first operations:
		#POS = (Mak.dot(np.multiply(d_MAin[idx,],G[idx,]))).dot(Malpha.dot(d_zd[idx]))		
		usual_matrix_mult1d(d_buffer2,d_buffer3, d_POS)

		#Here we start with NEG
		#np.diag(d_zd[idx]
		#zd diag
		d_zd_diag = cuda.local.array((9,9),dtype=types.float32)
		device_diag(d_zd[idx,], d_zd_diag)

		#-Malpha.dot(np.diag(d_zd[idx])		
		usual_matrix_mult(d_alpha_diag,d_zd_diag, d_buffer1)		
		
		#np.multiply(d_MAout[idx,],G[idx,])
		element_matrix_mult(idx,d_MAout,d_G, d_buffer2)



		#buffer3 save (np.multiply(d_MAout[idx,],G[idx,]).dot(np.diag(Mak)))
		#operation with Ak like vector no matrix, because
		#in original operation is a double diag

		usual_matrix_mult1d(d_buffer2,d_ak[idx,], d_buffer3)

		#Final operation to get NEG
		#NEG = -Malpha.dot(np.diag(d_zd[idx]).dot(np.multiply(d_MAout[idx,],G[idx,]).dot(np.diag(Mak))))
		usual_matrix_mult1d(d_buffer1,d_buffer3, d_NEG)

		for i in range(9):
			d_zd[idx,i] = d_POS[i]+d_NEG[i]


#To be called only from device
@cuda.jit(device=True)
def device_diag(original,diagonal):
	for x in range(original.shape[0]):
		for y in range(original.shape[0]):
			if x == y:
				diagonal[x,y] = original[x]			

#To be called only from device
#
@cuda.jit(device=True)
def element_matrix_mult(idx,matrix1,matrix2, result):
	for x in range(matrix1.shape[1]):
		for y in range(matrix1.shape[2]):
			result[x,y] = matrix1[idx,x,y]*matrix2[idx,x,y]

@cuda.jit(device=True)
def element_matrix_mult_buffer(matrix1,matrix2,result):
	for x in range(matrix1.shape[0]):
		for y in range(matrix1.shape[1]):
			result[x,y] = matrix1[x,y]*matrix2[x,y]

	
@cuda.jit(device=True)
def usual_matrix_mult(matrix1,matrix2,result):
	for x in range(matrix1.shape[0]):
		for y in range(matrix2.shape[1]):
			for z in range(matrix2.shape[0]):
				result[x,y] += matrix1[x,z]*matrix2[z,y]


@cuda.jit(device=True)
def usual_matrix_mult1d(matrix1,matrix2, result):
	#sum = 0.00
	for x in range(matrix1.shape[0]):
		for y in range(matrix2.shape[0]):
			result[x] += matrix1[x,y]*matrix2[y]


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


def pythonGPUIntegrate(request):
	#To measure time	
	start = time.perf_counter()
	initIntegrateModel()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')

	return render(request,'showmap.html')


def initIntegrateModel():
	
	#Bloques e hilos
	blocksPerGrid = 4
	threadsPerBlock = 4

	#Samples
	tspan = numpy.linspace(0,10,10) 
	
	arrayResults = np.zeros(1000)
	d_arrayResults =  cuda.to_device(arrayResults)

	z0 = np.array([2,3])
	#pdb.set_trace()
	
	integrateGPU[blocks_per_grid,threadsPerBlock](d_arrayResults)

	arrayResults = d_arrayResults.copy_to_host(d_arrayResults)

	plt.figure("Zp", dpi=150)

	plt.subplot(3,4,1)
	plt.xlabel('Time')
	plt.ylabel("z7")
	plt.plot(zps[:,6])



@cuda.jit
def integrateGPU():
	pass



