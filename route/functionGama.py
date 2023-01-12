initModel()


def initModel():
	
	#Simulation

	#deSPUÉS CADA SEGMENTO SU ALPHA MAX
	#Globar vars
	global ALPHA_MAX
	ALPHA_MAX =.04
	global ALPHA_MIN
	ALPHA_MIN = .01
	global RHO
	RHO = 1


	#Traffic lights
	global GREEN_TIME_VERTICAL
	GREEN_TIME_VERTICAL = 15
	global GREEN_TIME_HORIZONTAL
	GREEN_TIME_HORIZONTAL = 15
	global YELLOW_TIME
	YELLOW_TIME = 2
	global ALL_RED_TIME
	ALL_RED_TIME = 2
	global STEP_TIME
	STEP_TIME = .1


	currentStates=[]
	times=[]
	weight52s=[]
	weight54s=[]
	h15s=[]
	h35s=[]

	def zp(z,t,G):
		T=30
		A=5
		w=2*numpy.pi/T
		return G*A*np.sin(w*t)


	cycles = 1
	for cycle in range(cycles):
		print("Cycle", cycle)
		totalTime = GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME+GREEN_TIME_VERTICAL+YELLOW_TIME+ALL_RED_TIME
		currentState = 1		
		weight52 = 0.0
		weight54 = 0.0
		h15 = 0.0
		h35 = 0.0
		index = 0
		t = 0.0
		zinitial = 0
		#zs = np.zeros(int(totalTime/STEP_TIME))
		zs = []
		counter =0

		while(t<totalTime):
			currentState, time, weight52, weight54, h15, h35 = stateMachine(t,currentState)
			times.append(time)
			weight52s.append(weight52)
			weight54s.append(weight54)
			h15s.append(h15)
			h35s.append(h35)
			

			tspan =numpy.linspace(t,t+STEP_TIME,2)
			z = odeint(zp, zinitial,tspan,args=(h15,))
			#zs[counter] = z[-1]
			zs.append(z[-1])
			counter+=1
			t +=STEP_TIME
			zinitial = z[-1]

	
	
	plt.figure()
	plt.subplot(1,1,1)
	plt.xlabel('Time')
	plt.ylabel("zp")
	plt.plot(zs)


	plt.figure()
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

	return "Model initiated"



def stateMachine(time, currentState):
	weight52 = 0.0
	weight54 = 0.0
	h15 = 0.0
	h35 = 0.0 

	if (time>0 and time<(GREEN_TIME_HORIZONTAL+YELLOW_TIME)):
		currentState = 1
		weight52 = 0.6
		weight54 = 0.4
		h15 = 1
		h35 = 0
	elif (time>=(GREEN_TIME_HORIZONTAL+YELLOW_TIME) and 
		time<(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME)):
		currentState = 2
		weight52 = 0.6
		weight54 = 0.4
		h15 = 0
		h35 = 0
	elif (time>=(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME) 
		and (time<(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME+
			GREEN_TIME_VERTICAL+YELLOW_TIME))):
		currentState = 3
		weight52 = 0.4
		weight54 = 0.6
		h15 = 0
		h35 = 1
	elif (time>=(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME+YELLOW_TIME) 
		and (time<(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME+
			GREEN_TIME_VERTICAL+YELLOW_TIME+ALL_RED_TIME))):
		currentState = 4
		weight52 = 0.4
		weight54 = 0.6
		h15 = 0
		h35 = 0
	else:
		currentState = 1
		time=0
		weight52 = 0.6
		weight54 = 0.4
		h15 = 1
		h35 = 0

	return currentState,time, weight52, weight54, h15, h35
    

def alpha(segmentS):
	return ((ALPHA_MAX-ALPHA_MIN)/(RHO-segmentS.limitOfCars))
	(segmentS.currentCars-segmentS.limitOfCars)+ALPHA_MIN