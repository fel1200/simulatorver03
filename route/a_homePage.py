from django.shortcuts import render

#home page
def home(request):
	#Return page	
	return render(request, 'home.html')

