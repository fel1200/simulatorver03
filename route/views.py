
import folium
from django.shortcuts import render,redirect
from . import getroute

#create views here

def showmap(request):
    return render(request,'showmap.html')

def showroute(request,lat1,long1,lat2,long2):
	figure = folium.Figure()
    lat1,long1,lat2,long2=float(lat1),float(long1),float(lat2),float(long2)
    route=getroute.get_route(long1,lat1,long2,lat2)
    m = folium.Map(location=[(route['start_point'][0]),
                                 (route['start_point'][1])], 
                       zoom_start=10)
    m.add_to(figure)
    folium.PolyLine(route['route'],weight=8,color='blue',opacity=0.6).add_to(m)
    folium.Marker(location=route['start_point'],icon=folium.Icon(icon='play', color='green')).add_to(m)
    folium.Marker(location=route['end_point'],icon=folium.Icon(icon='stop', color='red')).add_to(m)
    figure.render()
    context={'map':figure}
    return render(request,'showroute.html',context)

def showmapSimulator(request):
	figure = folium.Figure()
	place_lat = [19.4320048, 19.4317964]
	place_lng = [-99.1412023, -99.1397023]
	m = folium.Map(location=[place_lat[0],	place_lng[0]],zoom_start=18)
	m.add_to(figure)
	folium.Marker(location=[place_lat[0],place_lng[0]], icon=folium.Icon(icon='info-sign', color='blue')).add_to(m)
	folium.Marker(location=[place_lat[1],place_lng[1]], icon=folium.Icon(icon='info-sign', color='blue')).add_to(m)

	points = []
	for i in range(len(place_lat)):
		points.append([place_lat[i], place_lng[i]])
		
	folium.PolyLine(points,weight=8,color='blue',opacity=0.6).add_to(m)
	figure.render()
	context={'map':figure}
	return render(request, 'showroute.html', context) 

def showmapInitial(request):
	"""It shows the initial map to interact with"""
	figure = folium.Figure()
	place_lat = [19.4320048, 19.4317964, 19.4316592]
	place_lng = [-99.1412023, -99.1397023, -99.1385719]
	#m = folium.Map(location=[place_lat[1],	place_lng[1]], zoom_start=18)

	#Getting data from openStreetMap
	m = folium.Map(location=[place_lat[1],
		place_lng[1]],
		tiles ='OpenStreetMap',
		zoom_start=18)
	

    	
    	


	#m = folium.Map(location=[45.372, -121.6972],
    #       zoom_start=12,
    #       tiles='http://{s}.tiles.yourtiles.com/{z}/{x}/{y}.png',
    #       attr='My Data Attribution')

	#Adding the map to the figure
	m.add_to(figure)
	
	#Markers
	folium.Marker(location=[place_lat[0],place_lng[0]], popup= "<i>Intersection</i>", icon=folium.Icon(icon='glyphicon-road', color='blue')).add_to(m)
	folium.Marker(location=[place_lat[1],place_lng[1]], popup= "<i>Intersection</i>", icon=folium.Icon(icon='glyphicon-road', color='blue')).add_to(m)
	folium.Marker(location=[place_lat[2],place_lng[2]], popup= "<i>Intersection with traffic light</i>", icon=folium.Icon(icon='screenshot', color='blue')).add_to(m)
	
	#To add interaction to show latitud and longitud
	m.add_child(folium.LatLngPopup())

	#To add on-the-fly marker
	#m.add_child(folium.ClickForMarker(popup="Waypoint"))
	
	points = []
	for i in range(len(place_lat)):
		points.append([place_lat[i], place_lng[i]])
		
	folium.PolyLine(points,weight=8,color='blue',opacity=0.6).add_to(m)

	figure.render()
	context={'map':figure}
	return render(request, 'showroute.html', context) 

