#To measure time
import time
#Arrays functionality
import numpy as np
#To debug
import pdb
from django.shortcuts import render
#To integrate
from scipy.integrate import odeint
from scipy import integrate
#To use math functions
import math
#To plot values
import matplotlib.pyplot as plt 


#Beginning method
def CPUArrays(request):
	#To measure time	
	start = time.perf_counter()
	initModelArray()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')

	return render(request,'showmap.html')


#settling model
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
	SUB_STEP_TIME = .01

	#times of vertical and horizontal		
	global MIN_GREEN_TIME_HORIZONTAL
	MIN_GREEN_TIME_HORIZONTAL = 20

	global MAX_GREEN_TIME_HORIZONTAL		
	MAX_GREEN_TIME_HORIZONTAL = 21

	global MIN_GREEN_TIME_VERTICAL		
	MIN_GREEN_TIME_VERTICAL = 20

	global MAX_GREEN_TIME_VERTICAL		
	MAX_GREEN_TIME_VERTICAL = 21

	global TOTAL_TIME
	TOTAL_TIME = 900

	#To plot or not
	isPlotting = True

	#Number of samples in zd
	samples = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME)	)
	
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

	#For all simulations including first sampling
	counterTotal = 0

	#To save all simulations including first sampling
	zpSimulations = np.empty((samples,12),dtype=float)

	for green_time_horizontal in range(MIN_GREEN_TIME_HORIZONTAL,MAX_GREEN_TIME_HORIZONTAL):
		for green_time_vertical in range(MIN_GREEN_TIME_VERTICAL,MAX_GREEN_TIME_VERTICAL):
			#print("Green_time_horizontal", green_time_horizontal)
			#print("Green_time_vertical", green_time_vertical)
			#totalTime of traffic light cycles
			totalTime = green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME			
			#pdb.set_trace()
			#To plot values
			weight52s = np.zeros((samples), dtype=float)
			weight54s = np.zeros((samples), dtype=float)
			h15s = np.zeros((samples), dtype=float)
			h35s = np.zeros((samples), dtype=float)
			alphas = np.zeros((9,samples), dtype=float)
			aks = np.zeros((9,samples), dtype=float)

			#To save alpha and ak
			alpha = np.zeros((9), dtype=float)
			ak = np.zeros((9), dtype=float)

			#Setting gammas in 0 for connection between segments
			G = np.zeros((9,9), dtype=float)


			#Initial vars for state machine
			currentState = 1
			weight52 = 0.0
			weight54 = 0.0
			h15 = 0.0
			h35 = 0.0

			#Empty array to save zp
			zps = np.empty((0,9), dtype=float)

			#time counter in cycle
			t=0.00
			counter = 0
			#CurrentCars
			zCars = np.array([0.0,0.0,0.0,0.0,0.0,100.0,
				100.0,0.0,0.0])

			#Obtaining z0, initial cars
			z0 = np.array([0.0,0.0,0.0,0.0,0.0,100.0,
				100.0,0.0,0.0])

			#Matrix definition
			MAout = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1],	
					[0, 1, 0, 1, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0]])

			#Transpose to obtain the output of segments
			MAin = MAout.T	
			

			#Matlab data to 1500 simulations, in python 900
			while(t<TOTAL_TIME): 
				#print("t",t)	
				tspan =np.linspace(t,t+STEP_TIME,int(STEP_TIME/SUB_STEP_TIME))
				#pdb.set_trace()					
				zp = odeint(updateCycle, z0, tspan,
				args=(alpha, ak,  alphas, aks, counter,
				currentState, time, weight52, 
				weight54, h15, h35, t, green_time_horizontal,
				green_time_vertical, weight52s, weight54s, 
				h15s, h35s, G, isPlotting, ZMAX, zCars, MAin, MAout))
				#pdb.set_trace()					
				#z0 is the last value of odeint
				z0 = zp[-1]

				#zpWithTrafficLightTime = np.array([green_time_horizontal,
				#	green_time_vertical, t])
				
				#zConcatenated = np.concatenate((zpWithTrafficLightTime,z0))
				#zpSimulations[counterTotal,] = zConcatenated

				#Saving data in zps	
				zps = np.append(zps,zp, axis=0)

				counter+=1
				counterTotal+=1
				t +=STEP_TIME
				#print(counter)
			counterSimulations +=1
			if(counterSimulations%10==0):
				print(counterSimulations)




	if (isPlotting):
		plot(zps, h15s, h35s, weight54s, weight52s)
		#np.savetxt("zps.csv", zps,delimiter = ',')
		#np.savetxt("zpSimulations.csv", zpSimulations,delimiter = ',')


def updateCycle(z0,timeS, alpha, ak, alphas, aks, counter,
					currentState, time, weight52, 
					weight54, h15, h35, t, 
					green_time_horizontal, green_time_vertical,
					weight52s, weight54s, h15s, h35s, G, isPlotting, ZMAX, zCars, MAin, MAout):
	#print("Enter", timeS)
#	pdb.set_trace()					
	for i in range(0,9):
		if z0[i]>ZMAX[i]:
			#pdb.set_trace()		
			z0[i]=ZMAX[i]
		elif z0[i] < 0.00:
			z0[i] = 0.00
	#Saving z0 as the new zd		
	
	zd = z0.copy()

	#Update alpha and ak in each value
	updateAlphaAk(alpha, ak, alphas,aks,counter, zd, zCars, isPlotting)
	#pdb.set_trace()		
	#Matrix diagonal of alpha and ak
	Malpha= np.diag(alpha.T)
	Mak= np.diag(ak.T)

	#To know the state of cycle
	div = int(timeS/(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME))
	timeSM = timeS-    (div * (green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME))
	currentState, time, weight52, weight54, h15, h35 = stateMachine(timeSM,currentState, green_time_horizontal, green_time_vertical, timeS)

	#Updating gamma
	updateG(G,h15,h35,weight52,weight54)
	#To update after a complete cycle and don't mix data
	zd_temp = zd.copy()
	zd=updateInputOutput(G, zd, timeS, zd_temp, Malpha, Mak, MAin, MAout)
#	pdb.set_trace()		
	return zd

def updateInputOutput(G, zd, time, zd_temp, Malpha, Mak, MAin, MAout):


	
	POS = (Mak.dot(np.multiply(MAin,G))).dot(Malpha.dot(zd_temp))
	#NEG = -np.diag(Mak).dot(np.multiply(MAout,G)).dot(Malpha.dot(np.diag(zd_temp)))
	NEG = -Malpha.dot(np.diag(zd_temp).dot(np.multiply(MAout,G).dot(np.diag(Mak))))
	zd = POS+NEG
	#pdb.set_trace()		
	return zd 	
	


	#for key in Segments:
	#	Segment = Segments[key]
	#	zd[Segment.numberOfSegment-1] = Segment.currentCars





	
def updateAlphaAk(alpha, ak, alphas,aks,counter,zd, zCars, isPlotting):
	for i in range(0,9):
		alpha[i] = getAlpha(i,zd, zCars)
		ak[i] = getAk(i,zd)
		if isPlotting:
			alphas[i,counter] = alpha[i]
			aks[i,counter] = ak[i]
		#inital values of zCars
		#zCars[Segment.numberOfSegment-1,counter] = Segment.nuco	pdb.set_trace()
		#pdb.set_trace()						

def getAlpha(i, zd, zCars):
	alpha = 0.00
	if (zd[i] < RHOA[i]):
		alpha = ALPHA_MAX
	elif ((zd[i] > RHOA[i]) and 
		(zd[i] <= ZMAX[i])):
		alpha = ((ALPHA_MAX-ALPHA_MIN)/(RHOA[i]-ZMAX[i]))*(zCars[i]-ZMAX[i])+ALPHA_MIN
	elif ((zd[i]>ZMAX[i])):
		alpha = ALPHA_MIN
	return alpha

def getAk(i, zd):
	ak = 1.0
	zk = zd[i]
	delta = (15*((zk/ZMAX[i]) - 0.7))
	ei =math.exp(delta)
	ak = 1/(1+ei)
	return ak

def stateMachine(time, currentState, green_time_horizontal, green_time_vertical,timeSimulation):
	weight52 = 0.0
	weight54 = 0.0
	h15 = 0.0
	h35 = 0.0 
	#Horizontal time
	if (time>0 and time<(green_time_horizontal+YELLOW_TIME)):
		currentState = 1
		weight52 = 0.6
		weight54 = 0.4
		h15 = 1
		h35 = 0
	elif (time>=(green_time_horizontal+YELLOW_TIME) and 
		time<(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME)):
		currentState = 2
		weight52 = 0.6
		weight54 = 0.4
		h15 = 0
		h35 = 0
	elif (time>=(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME) 
		and (time<(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+
			green_time_vertical+YELLOW_TIME))):
		currentState = 3
		weight52 = 0.4
		weight54 = 0.6
		h15 = 0
		h35 = 1
	elif (time>=(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+YELLOW_TIME) 
		and (time<(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+
			green_time_vertical+YELLOW_TIME+ALL_RED_TIME))):
		currentState = 4
		weight52 = 0.4
		weight54 = 0.6
		h15 = 0
		h35 = 0
	else:
		currentState = 1
		#time=0
		weight52 = 0.6
		weight54 = 0.4
		h15 = 1
		h35 = 0
	#if timeSimulation>1000:
		#pdb.set_trace()			
	#print("time:", time, " w52:", weight52, "w54:", weight54, "h15:", h15, "h35:", h35)	
	return currentState,time, weight52, weight54, h15, h35

def updateG(G,h15,h35,weight52,weight54):
	#Defining gammas for intersection connections
	G[0,4]= G[4,0]= h15 #Gamma for connection 1,5
	G[2,4]= G[4,2]=h35 #Gamma for connection 3,5
	G[4,1]= G[1,4]=weight52 #Gamma weight for connection 5,2
	G[4,3]= G[3,4]=weight54 #Gamma for connection 5,4
		
	G[1,7]= G[7,1]= 1 #Gamma for connection 2,8
	G[3,8]= G[8,3]= 1 #Gamma for connection 4,9
	G[5,0]= G[0,5]= 1 #Gamma for connection 6,1
	G[6,2]= G[2,6]= 1 #Gamma for connection 7,3
	#pdb.set_trace()			


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
		plt.plot(weight54s)

		plt.show()






