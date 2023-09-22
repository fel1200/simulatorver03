from django.shortcuts import render

#home page with html



def home(request):
	#Return page	
	#return render(request,'showmap.html')
	return render(request, 'home.html')

