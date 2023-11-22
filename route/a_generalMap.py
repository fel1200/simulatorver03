#Folium to draw maps, getroute from this project to get a from two points defined
import folium
from django.shortcuts import render,redirect
from django.http import HttpResponse

from . import a_getroute

#code to fetch data from openstreet
import osmnx as ox

#Import to drawing
from shapely.geometry import LineString, MultiLineString, Point

#Import pandas for some traffic utilites
import pandas as pd


#To calculate distance
from math import radians, cos, sin, asin, sqrt

#to check GPU
import tensorflow as tf
from tensorflow.python.client import device_lib

#To numpy functions
import numpy


#To debug
import pdb

#utilities
import xml.dom.minidom 


#create views here
#To parse info from map
def parse(request):
    #parse the information of the road
    xmldoc = xml.dom.minidom.parse('mapComplexMex.osm')
    nodes = xmldoc.getElementsByTagName('node')
    ways = xmldoc.getElementsByTagName('way')
    message = str(len(nodes))+' nodes,'+ str(len(ways))+ ' ways' 
    return HttpResponse(message)

def showmapRandom(request):
    return render(request,'showmap.html')

def showRoute(request,lat1,long1,lat2,long2):
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

def showmapMexicoCityPoints(request):
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

def showmapSelector(request, selector):
	"""It shows the initial map to interact with"""
	figure = folium.Figure()

	"""Selector 
		1 = Mexico City, simple crossing
		2 = Mexico City, medium square
		3 = Mexico City, large square
		4 = Av. Gral. Garibaldi, Montevideo	
	"""
	if (selector == 1):
		graph = ox.graph_from_bbox(19.43323, 19.42950,-99.14130,-99.13328, network_type='drive')
		m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=25)
	elif (selector == 2):
		graph = ox.graph_from_bbox(19.4362, 19.4244,-99.1478,-99.1279, network_type='drive')
		m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=25)
	elif (selector == 3):
		graph = ox.graph_from_bbox(19.4451, 19.4216,-99.1637,-99.1238, network_type='drive')
		m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=15)
	elif (selector == 4):
		graph = ox.graph_from_bbox(-34.88791, -34.89025, -56.16213, -56.15824, network_type='drive')
		m = folium.Map(location=[-34.88948,-56.16025], zoom_start=25)
	
	#converter to gdfs
	nodes, edges = ox.graph_to_gdfs(graph)
	style = {'color': '#F7DC6F', 'weight':'1'}
	df = pd.DataFrame(edges).apply(tuple)
	df.to_csv('streets.csv', index=False)
	streetsaslinestring = edges.loc[: , "geometry"]
	edges_series = edges['length'] #gives you a pandas series with edge lengths
	numberEdges = len(graph.edges)

	points = []
	for numberEdge in range(0,numberEdges-1):
		line = list(edges.geometry.iloc[numberEdge].coords)		
		counter = 0
		for pt in line:
			points.append(pt)
			if counter == 0:
				coordLat1 = Point(pt).y
				coordLong1 = Point(pt).x
			else:
				coordLat2 = Point(pt).y
				coordLong2 = Point(pt).x
			counter += 1
		coordinates = [[0 for i in range(2)] for i in range(2)]
		coordinates[0][0] = coordLat1
		coordinates[0][1] = coordLong1	
		coordinates[1][0] = coordLat2
		coordinates[1][1] = coordLong2
		
		#Half between two points
		lonA = math.radians(coordLong1)
		lonB = math.radians(coordLong2)
		latA = math.radians(coordLat1)
		latB = math.radians(coordLat2)
		dLon = lonB - lonA
		#print(dLon)
		Bx = math.cos(latB) * math.cos(dLon)
		By = math.cos(latB) * math.sin(dLon)

		latC = math.atan2(math.sin(latA) + math.sin(latB),math.sqrt((math.cos(latA)+Bx) * (math.cos(latA)+Bx) + By * By))
		lonC = lonA + math.atan2(By,math.cos(latA) + Bx)  
		lonC = (lonC +3 * math.pi) % (2 * math.pi) - math.pi
		lonHalf = math.degrees(lonC)
		latHalf = math.degrees(latC)
		
		#To calculate distance
		segmentLenght = (haversine(coordLat1,coordLong1,coordLat2,coordLong2)*1000)-10
		textToMarker = "Street segment lenght: "+str(segmentLenght) + " meters"

		folium.Marker(location=[latHalf,lonHalf],popup= textToMarker,icon=folium.Icon(icon='glyphicon-road',color='gray')).add_to(m)
			
		folium.PolyLine(locations=coordinates, line_color='#FF0000', line_weight=5).add_to(m)
	folium.GeoJson(edges.sample(), style_function=lambda x: style).add_to(m)
	m.add_to(figure)

	idNodes = list(graph.nodes)
	for idNode in list(graph.nodes):
		latNode=graph.nodes[idNode]['y'] #lat
		longNode=graph.nodes[idNode]['x'] #lon
		textToMarker = "Intersection lat: "+ str(latNode) + " long : "+ str(longNode)
		folium.Marker(location=[latNode
			,longNode], 
			popup= textToMarker ,
			icon=folium.Icon(icon='glyphicon-th-large',
			 color='blue')).add_to(m)

	figure.render()
	context={'map':figure}

	return render(request, 'showroute.html', context) 

   


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    #r = 6372.8 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r




	
