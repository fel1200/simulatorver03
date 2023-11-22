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

#Our model of streets
from route.models import SegmentStreet

#begining function
def CPUClasses(request):
	
	#Timer to measure how it takes to finish
	start = time.perf_counter()
	initModelClasses()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start,2)} second(s)')
	return render(request,'showmap.html')


#init and execution of the modl
def initModelClasses():

	
	#Constants and global values
	RHO_SEGMENT	= .02

	#Globar vars
	global ALPHA_MAX
	ALPHA_MAX =.04
	global ALPHA_MIN
	ALPHA_MIN = .01
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
	MIN_GREEN_TIME_HORIZONTAL = 10

	global MAX_GREEN_TIME_HORIZONTAL		
	MAX_GREEN_TIME_HORIZONTAL = 12

	global MIN_GREEN_TIME_VERTICAL		
	MIN_GREEN_TIME_VERTICAL = 10

	global MAX_GREEN_TIME_VERTICAL		
	MAX_GREEN_TIME_VERTICAL = 12

	global TOTAL_TIME
	TOTAL_TIME = 900
		
	#Objects' declaration
	"""self, name, numberOfSegment, type_of_segment, orientation, initialCars,
		currentCars, tempCurrentCars, limitOfCars, inputs, , z, zp
		orientation_inputs,
		outputs, orientation_outputs, rho, alpha, ak"""

	B1 = SegmentStreet("B1",1,"ST","HO",0,0,0,35,['B6'],['HO'],['B5'],['IN'], 35*RHO_SEGMENT,0.0,0.0,0.0,0.0)
	B2 = SegmentStreet("B2",2,"ST","HO",0,0,0,35,
		['B5'],['HO'],['B8'],['HO'], 35*RHO_SEGMENT, 0.0,0.0,0.0,0.0)
	B3 = SegmentStreet("B3",3,"ST","VE",0,0,0,35,
		['B7'],['VE'],['B5'],['VE'], 35*RHO_SEGMENT, 0.0,0.0,0.0,0.0)
	B4 = SegmentStreet("B4",4,"ST","VE",0,0,0,35,
		['B5'],['VE'],['B9'],['VE'], 35*RHO_SEGMENT, 0.0,0.0,0.0,0.0)
	B5 = SegmentStreet("B5",5,"IT","IN",0,0,0,10,
		['B1','B3'],['HO','VE'],['B2','B4'],['HO','VE'], 10*RHO_SEGMENT,
 		0.0,0.0,0.0,[])
	B6 = SegmentStreet("B6",6,"SO","HO",100,100,100,200,
		[],[],['B1'],['HO'], 200*RHO_SEGMENT, 0.0,0.0,0.0,100)
	B7 = SegmentStreet("B7",7,"SO","VE",100,100,100,200,
		[],[],['B3'],['VE'], 200*RHO_SEGMENT, 0.0,0.0,0.0,100)
	B8 = SegmentStreet("B8",8,"SI","HO",0,0,0,200,
		['B2'],['HO'],[],[], 200*RHO_SEGMENT, 0.0,0.0,0.0,0.0)
	B9 = SegmentStreet("B9",9,"SI","VE",0,0,0,200,
		['B4'],['VE'],[],[], 200*RHO_SEGMENT, 0.0,0.0,0.0,0.0)
	
	#In each iteration we need to restore the value of currentCars
	#for every segment
	initialCars = np.array([0,0,0,0,0,100,100,0,0])
	Segments = {B1.name:B1,
				B2.name:B2,
				B3.name:B3,
				B3.name:B3,
				B4.name:B4,
				B5.name:B5,
				B6.name:B6,
				B7.name:B7,
				B8.name:B8,
				B9.name:B9,
				}

	#To print charts			
	isPlotting = True

	#number Of simulations
	simulations = (MAX_GREEN_TIME_HORIZONTAL-MIN_GREEN_TIME_HORIZONTAL)*(MAX_GREEN_TIME_VERTICAL-MIN_GREEN_TIME_VERTICAL)

	#total samples in integration
	samples = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME)	)


	carsDischarged = np.zeros(simulations, dtype=float)	

	counterSimulations = 0

	#For all simulations including first sampling
	counterTotal = 0

	#zpSimulations To save complete array
	#First number: first traffic light
	#Second number: second traffic light
	#Third number: overall timeSimulation
	#Fourth to 12th number: data for each segment
	zpSimulations = np.empty((samples,12),dtype=float)



	#simulations cycle
	for green_time_horizontal in range(MIN_GREEN_TIME_HORIZONTAL,MAX_GREEN_TIME_HORIZONTAL):
		for green_time_vertical in range(MIN_GREEN_TIME_VERTICAL,MAX_GREEN_TIME_VERTICAL):

			#Values of Zmax in each segment
			ZMAX = np.zeros((9), dtype=float)
			for key in Segments:
				Segment = Segments[key]
				Segment.currentCars = initialCars[Segment.numberOfSegment-1]
				Segment.initialCars = initialCars[Segment.numberOfSegment-1]
				Segment.tempCurrentCars = initialCars[Segment.numberOfSegment-1]
				ZMAX[Segment.numberOfSegment-1]= Segment.limitOfCars


			#To save data of weight and h
			weight52s = np.zeros((samples), dtype=float)
			weight54s = np.zeros((samples), dtype=float)
			h15s = np.zeros((samples), dtype=float)
			h35s = np.zeros((samples), dtype=float)
			
			
			#To save values of alphas and aks
			alphas = np.zeros((9,samples), dtype=float)
			aks = np.zeros((9,samples), dtype=float)

			#Setting gammas in 0 for connection between segments
			G = np.zeros((9,9), dtype=float)

			#Initial vars for state machine
			currentState = 1
			weight52 = 0.0
			weight54 = 0.0
			h15 = 0.0
			h35 = 0.0
	
			#Initializating zps
			zps = np.empty((0,9), dtype=float)

			#to save time in each simulation
			t=0.00
			
			#counter for cycles
			counter = 0


			#Obtaining z0, initial cars
			z0 = np.zeros((len(Segments)), dtype=float)
			for key in Segments:
				Segment = Segments[key]
				z0[Segment.numberOfSegment-1] = Segment.currentCars

			#pdb.set_trace()	

			while(t<TOTAL_TIME): 
				#Samples each iteration
				tspan =np.linspace(t,t+STEP_TIME,int(STEP_TIME/SUB_STEP_TIME))

				#Update in integration value
				zp = odeint(updateCycle, z0, tspan,
				args=(Segments, alphas, aks, counter,
				currentState, time, weight52, 
				weight54, h15, h35, t, green_time_horizontal,
				green_time_vertical, weight52s, weight54s, 
				h15s, h35s, G, isPlotting, ZMAX))

				#Saving last value in array	to use it in next cycle			
				z0 = zp[-1]	

				#To save all values of zps
				zps = np.append(zps,zp, axis=0)

				#pdb.set_trace()	
				counter+=1
				counterTotal+=1
				t +=STEP_TIME
				#pdb.set_trace()	
				#print("Sampling:",counter)
			print("Simulation:",counterSimulations)
			counterSimulations +=1
		
	if (isPlotting):
		plot(zps, h15s, h35s, weight54s, weight52s)
		#np.savetxt("zpS.csv", zps,delimiter = ',')
		np.savetxt("zpSimulations.csv", zpSimulations,delimiter = ',')



	
def updateCycle(z0,timeS,Segments, alphas, aks, counter,
					currentState, time, weight52, 
					weight54, h15, h35, t, 
					green_time_horizontal, green_time_vertical,
					weight52s, weight54s, h15s, h35s, G, isPlotting, ZMAX):
	
	#Checking max values
	for i in range(0,len(Segments)):
		if z0[i]>ZMAX[i]:
			#pdb.set_trace()		
			z0[i]=ZMAX[i]
		elif z0[i] < 0.00:
			z0[i] = 0.00

	#Copy of z values
	zd = z0.copy()
	
	#Function to get alpha and ak values
	updateAlphaAk(Segments,alphas,aks,counter, zd)

	#Getting data from stateMachine
	div = int(timeS/(green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME))
	timeSM = timeS-    (div * (green_time_horizontal+YELLOW_TIME+ALL_RED_TIME+green_time_vertical+YELLOW_TIME+ALL_RED_TIME))
	currentState, time, weight52, weight54, h15, h35 = stateMachine(timeSM,currentState, green_time_horizontal, green_time_vertical, timeS)
	numberOfIterations = int(timeS/SUB_STEP_TIME)
	
	#Updating G values
	updateG(G,h15,h35,weight52,weight54)

	zd_temp = zd.copy()
	updateInputOutput(Segments,G, zd, timeS, zd_temp)

	return zd
	
def updateInputOutput(Segments,G, zd, time, zd_temp):
	difference = 0
	for key in Segments:
		Segment = Segments[key]
		u_input = 0.00
		y_output = 0.00
		if(Segment.inputs):
			for keyInput in Segment.inputs:
				SegmentInput = Segments[keyInput]
				u_input += SegmentInput.alpha*Segment.ak*zd_temp[SegmentInput.numberOfSegment-1]*G[SegmentInput.numberOfSegment-1,Segment.numberOfSegment-1]
		if(Segment.outputs):
			for keyOutput in Segment.outputs:
				SegmentOutput = Segments[keyOutput]
				y_output += Segment.alpha*SegmentOutput.ak*zd_temp[Segment.numberOfSegment-1]*G[Segment.numberOfSegment-1,SegmentOutput.numberOfSegment-1]
				if(math.isnan(y_output)):
					print("Not a number")

		difference = u_input-y_output
		Segment.currentCars = difference 
		zd[Segment.numberOfSegment-1] = Segment.currentCars
	

    
def updateAlphaAk(Segments,alphas,aks,counter,zd):
	for key in Segments:
		Segment = Segments[key]
		Segment.alpha = alpha(Segment,zd)
		Segment.ak = ak(Segment,zd)
		alphas[Segment.numberOfSegment-1,counter] = Segment.alpha
		aks[Segment.numberOfSegment-1,counter] = Segment.ak 


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


def alpha(segmentS, zd):
	alpha = 0.00
	if (zd[segmentS.numberOfSegment-1] < segmentS.rho):
		alpha = ALPHA_MAX
	elif ((zd[segmentS.numberOfSegment-1] > segmentS.rho) and 
		(zd[segmentS.numberOfSegment-1] <= segmentS.limitOfCars)):
		alpha = ((ALPHA_MAX-ALPHA_MIN)/(segmentS.rho-segmentS.limitOfCars))*(segmentS.currentCars-segmentS.limitOfCars)+ALPHA_MIN
#		alpha = ((ALPHA_MAX-ALPHA_MIN)/(segmentS.rho-zd[segmentS.numberOfSegment-1]))*(zd[segmentS.numberOfSegment-1]-segmentS.limitOfCars)+ALPHA_MIN
	elif ((zd[segmentS.numberOfSegment-1]>segmentS.limitOfCars)):
		alpha = ALPHA_MIN
	return alpha
	
	
def ak(segmentS, zd):
	ak = 1.0
	zk = zd[segmentS.numberOfSegment-1]
	zmax = segmentS.limitOfCars
	delta = (15*((zk/zmax) - 0.7))
	#print("delta:",delta)
	ei =math.exp(delta)
	ak = 1/(1+ei)
	#return round(ak,4)
	return ak



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

