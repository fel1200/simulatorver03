from django.shortcuts import render

#home page
def simulator(request):
	#Return page	
	return render(request, 'simulator.html')

