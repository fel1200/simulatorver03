INPUT_BLOCKS = {'B1' : ['B6'],
                        'B2' : ['B5'],
                        'B3' : ['B7'],
                        'B4' : ['B5'],
                        'B5' : ['B1','B3'],
                        'B6' : [''],
                        'B7' : [''],
                        'B8' : ['B2'],
                        'B9' : ['B4']
                                        }
        OUTPUT_BLOCKS = {'B1' : ['B5'],
                         'B2' : ['B8'],
                         'B3' : ['B5'],
                         'B4' : ['B9'],
                         'B5' : ['B2','B4'],
                         'B6' : ['B1'],
                         'B7' : ['B3'],
                         'B8' : [''],
                         'B9' : ['']
                          }

        INITIAL_CARS = {'B1' : 0,
                        'B2' : 0,
                        'B3' : 0,
                        'B4' : 0,
                        'B5' : 0,
                        'B6' : 100,
                        'B7' : 100,
                        'B8' : 0,
                        'B9' : 0
                         }
        MAX_CARS = {'B1' : 50,
                    'B2' : 50,
                    'B3' : 50,
                    'B4' : 50,
                    'B5' : 15,
                    'B6' : 200,
                    'B7' : 200,
                    'B8' : 200,
                    'B9' : 200
                     }                              
        RHO =    {'B1' :0,
                  'B2' : 0,
                  'B3' : 0,
                  'B4' : 0,
                  'B5' : 0,
                  'B6' : 0,
                  'B7' : 0,
                  'B8' : 0,
                  'B9' : 0
                  }
        ALPHA_MAX ={'B1' :1,
                    'B2' : 1,
                    'B3' : 1,
                    'B4' : 1,
                    'B5' : 1,
                    'B6' : 1,
                    'B7' : 1,
                    'B8' : 1,
                    'B9' : 1
                     }

        ALPHA_MIN =      {'B1' :1,
                                        'B2' : 1,
                                        'B3' : 1,
                                        'B4' : 1,
                                        'B5' : 1,
                                        'B6' : 1,
                                        'B7' : 1,
                                        'B8' : 1,
                                        'B9' : 1
                                        }
        KE =                     {'B1' :1,
                                        'B2' : 1,
                                        'B3' : 1,
                                        'B4' : 1,
                                        'B5' : 1,
                                        'B6' : 1,
                                        'B7' : 1,
                                        'B8' : 1,
                                        'B9' : 1
                                        }
        CHI ={'B1' :1,
              'B2' : 1,
              'B3' : 1,
              'B4' : 1,
              'B5' : 1,
              'B6' : 1,
              'B7' : 1,
              'B8' : 1,
              'B9' : 1
               }                                      
       
        
		Z = {'B1' :0,
             'B2' : 0,
             'B3' : 0,
             'B4' : 0,
             'B5' : 0,
             'B6' : 100,
             'B7' : 100,
             'B8' : 0,
             'B9' : 0
              } 

              for block in RHO:
                RHO[block] = MAX_CARS[block] * 0.2
                ALPHA_MAX[block] = ALPHA_MAX[block]*.04
                ALPHA_MAX[block] = ALPHA_MAX[block]*.01
                KE[block] = KE[block]*15
                CHI[block] = CHI[block]*0.7
               

                

     
       for z in Z:
                inputBlocks = INPUT_BLOCKS[z]
                outputBlocks = OUTPUT_BLOCKS[z]
                print("z=",z)
                print("input=",inputBlocks)
                print("output=",outputBlocks)
                #Z[z]= inputs()-outputs()
        


        for z in Z:

        """
        print("Inputs")
        for block in INPUT_BLOCKS:
                print(block)
                print(INPUT_BLOCKS[block])
       
        print("Outputs")
        for block in OUTPUT_BLOCKS:
                print(block)
                print(OUTPUT_BLOCKS[block])
                """    


                #Blocks' inputs
        INPUT_BLOCKS = 
        {'B1' : ['B6'],
        'B2' : ['B5'],
        'B3' : ['B7'],
        'B4' : ['B5'],
        'B5' : ['B1','B3'],
        'B6' : [''],
        'B7' : [''],
        'B8' : ['B2'],
        'B9' : ['B4']
        }
        
        #Blocks' output
        OUTPUT_BLOCKS =
        {'B1' : ['B5'],
        'B2' : ['B8'],
        'B3' : ['B5'],'B4' : ['B9'],
        'B5' : ['B2','B4'],'B6' : ['B1'],
        'B7' : ['B3'],'B8' : [''],'B9' : ['']
        }
        
        #Number of cars in each block
        INITIAL_CARS = {'B1' : 0,
                        'B2' : 0,
                        'B3' : 0,
                        'B4' : 0,
                        'B5' : 0,
                        'B6' : 100,
                        'B7' : 100,
                        'B8' : 0,
                        'B9' : 0
                         }
        MAX_CARS = {'B1' : 50,
                    'B2' : 50,
                    'B3' : 50,
                    'B4' : 50,
                    'B5' : 15,
                    'B6' : 200,
                    'B7' : 200,
                    'B8' : 200,
                    'B9' : 200
                     }                              
        RHO =    {'B1' :0,
                  'B2' : 0,
                  'B3' : 0,
                  'B4' : 0,
                  'B5' : 0,
                  'B6' : 0,
                  'B7' : 0,
                  'B8' : 0,
                  'B9' : 0
                  }
        ALPHA_MAX ={'B1' :1,
                    'B2' : 1,
                    'B3' : 1,
                    'B4' : 1,
                    'B5' : 1,
                    'B6' : 1,
                    'B7' : 1,
                    'B8' : 1,
                    'B9' : 1
                     }

        ALPHA_MIN =      {'B1' :1,
                                        'B2' : 1,
                                        'B3' : 1,
                                        'B4' : 1,
                                        'B5' : 1,
                                        'B6' : 1,
                                        'B7' : 1,
                                        'B8' : 1,
                                        'B9' : 1
                                        }
        KE =                     {'B1' :1,
                                        'B2' : 1,
                                        'B3' : 1,
                                        'B4' : 1,
                                        'B5' : 1,
                                        'B6' : 1,
                                        'B7' : 1,
                                        'B8' : 1,
                                        'B9' : 1
                                        }
        CHI ={'B1' :1,
              'B2' : 1,
              'B3' : 1,
              'B4' : 1,
              'B5' : 1,
              'B6' : 1,
              'B7' : 1,
              'B8' : 1,
              'B9' : 1
               }                                      
       
        
                Z = {'B1' :0,
             'B2' : 0,
             'B3' : 0,
             'B4' : 0,
             'B5' : 0,
             'B6' : 100,
             'B7' : 100,
             'B8' : 0,
             'B9' : 0
              } 

       
        return ""       




"""
#cycle with TGV, TGH
    for seconds in range(50):
        z1 = B1.inputs.value - B1.ouputs.values
            alpha()ak()Gamma()z1-



def gamma(seconds):
        if seconds < tAR:
            return 0
    maquinaEsta

def funcCPU:
    ....    
kernel(TGV):
    self, name, type_of_segment, initialCars,
        currentCars, limitOfCars, alphaValue, inputs, outputs)"""




        """elif (time>=(GREEN_TIME_HORIZONTAL+YELLOW_TIME+ALL_RED_TIME+
            GREEN_TIME_VERTICAL+YELLOW_TIME+ALL_RED_TIME)):
        currentState = 1
        time=0
        weight52 = 0.6
        weight54 = 0.4
        h15 = 1
        h35 = 0"""
#Ponerlo en forma matricial
    def zp(z,t,G):
        
        T=30
        A=5
        w=2*numpy.pi/T
        return G*A*np.sin(w*t)


        
    for key in Segments:
        Segment = Segments[key]
        inputSum = 0
        outputSum = 0
        for keyInputSegment in Segment.inputs:
            #print("Checking segment ", Segment.name)
            if keyInputSegment:  #Check if array is not empty
                #print("Segment input ",keyInputSegment)
                inputSegment = Segments[keyInputSegment]            
                #Obtain values of "u" function (input)
                inputSum = alpha(inputSegment)
                #print(inputSegment.name, inputSum) 
        
        zp = inputSum
        #print(Segment.name,"-", zp)                 
        #To do
    #Solo es con salidas
    currentStates=[]
    times=[]
    weight52s=[]
    weight54s=[]
    h15s=[]
    h35s=[]

    #Ponerlo en forma matricial
    def zp(z,t,G):
        
        T=30
        A=5
        w=2*numpy.pi/T
        return G*A*np.sin(w*t)


    cycles = 1
    for cycle in range(cycles):
        #print("Cycle", cycle)
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
            #z = odeint(zp, zinitial,tspan,args=(h15,))
            #zs[counter] = z[-1]
            zs.append(z[-1])
            counter+=1
            t +=STEP_TIME
            zinitial = z[-1]

    
    if (isPlotting):
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

        plt.subplot(2,1,1)
        plt.xlabel('Time')
        plt.ylabel("Alpha")
        plt.plot(alphas)

        plt.subplot(2,1,2)
        plt.xlabel('Time')
        plt.ylabel("Ak")
        plt.plot(aks)

        #Getting precedence
    #This process can be done with type_of_Segment or with outputs
    Sinks = {}
    for key in Segments:
        Segment = Segments[key]
        if (not Segment.outputs): #Only interested in segment whitout exit connection (i.e. sinks)
            Sinks[Segment.name] = Segment
    
    #def zp(z,t,currentAlpha,currentAk,lastZ, weight, nextAlpha, nextAk, nextZ, h, nextnextAk):
    #   return nextAlpha*nextAk*weight*lastZ - currentAlpha*nextnextAk*h*currentZ


#AQUÍ EMPIEZA EL RESPALDO DE VIEWS


#Folium to draw maps, getroute from this project to get a from two points defined
import folium
from django.shortcuts import render,redirect
from . import getroute

#code to fetch data from openstreet
import osmnx as ox

#Import to drawing
from shapely.geometry import LineString, MultiLineString, Point

#Import pandas for some traffic utilites
import pandas as pd

#To use math functions
import math

#To calculate distance
from math import radians, cos, sin, asin, sqrt

#to check GPU
import tensorflow as tf
from tensorflow.python.client import device_lib

#To numpy functions
import numpy

#To plot values
import matplotlib.pyplot as plt 

#To integrate
from scipy.integrate import quad, odeint, solve_ivp
from scipy import integrate

#To debug
import pdb

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
    m = folium.Map(location=[place_lat[0],  place_lng[0]],zoom_start=18)
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

    #place = "Liverpool, United Kingdom"
    #graph = ox.graph_from_place(place, network_type='drive')
    #graph= ox.graph_from_point((37.79, -122.41), dist=750, network_type="drive")
    

    """Selector 
        1 = Mexico City
        2 = Av. Gral. Garibaldi, Montevideo 
    """
    selector = 1
    if selector ==1:
        #biggest selection on CDMX
        graph = ox.graph_from_bbox(19.4451, 19.4216,-99.1637,-99.1238, network_type='drive')
        m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=15)

        #bigger selection on CDMX
        #graph = ox.graph_from_bbox(19.4362, 19.4244,-99.1478,-99.1279, network_type='drive')
        #m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=25)

        #simple intersection on CDMX
        #graph = ox.graph_from_bbox(19.43323, 19.42950,-99.14130,-99.13328, network_type='drive')
        #m = folium.Map(location=[19.4316592,-99.1385719], zoom_start=25)

    else:
        graph = ox.graph_from_bbox(-34.88791, -34.89025, -56.16213, -56.15824, network_type='drive')
        m = folium.Map(location=[-34.88948,-56.16025], zoom_start=25)
    
    #converter to gdfs
    nodes, edges = ox.graph_to_gdfs(graph)
    #print(edges.head())
    #print(nodes.head())
    #streets.head()
    #route_line = edges['geometry'].tolist()
    #print(route_line)
    style = {'color': '#F7DC6F', 'weight':'1'}
    #m = folium.Map(location=[-2.914018, 53.366925],
    #zoom_start=15,
    #tiles="CartoDb dark_matter")
    df = pd.DataFrame(edges).apply(tuple)
    df.to_csv('streets.csv', index=False)
    streetsaslinestring = edges.loc[: , "geometry"]
    
    #lineLong = line.point_object.x
    #lineLat = line.point_object.y
    #print(lines)
    
    #to get lenghts
    edges_series = edges['length'] #gives you a pandas series with edge lengths
    #print(edges_series)

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
    #lines = list(edges.geometry.iloc[0].coords)
    
    #route=getroute.get_route(coordLong1,coordLat1,coordLong2,coordLat2)
    #folium.PolyLine(route['route'],weight=8,color='blue',opacity=0.6).add_to(m)
    #folium.PolyLine(points,weight=8,color='blue',opacity=0.6).add_to(m)    
    
    #coordinates = [
    #[19.4320048,   -99.1412023],
    #[19.4317964,   -99.1397023]]
    

    

    #print(lineLong)
    #print(lineLat)

    #print(streetsaslinestring)
    #line = streetsaslinestring[:0]
    #print(line)
    #lines = LineString(streetsaslinestring)
    #Lats, Lons = streetsaslinestring.coords.xy
    #print(Lats)
    #print(Lons)
    folium.GeoJson(edges.sample(), style_function=lambda x: style).add_to(m)
    #m.save("edges.html")
    m.add_to(figure)

    #getting and drawing nodes as markers
    idNodes = list(graph.nodes)
    #print(idNodes)
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


    ##GPU check version
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #print(device_lib.list_local_devices())

    #check_cuda()

    tx = cuda.threadIdx.x
    #ty = cuda.blockIdx.x
    #bw = cuda.blockDim.x
    #print("thread:",tx,"block:",ty,"thread:",bw)
    print("Thread",tx)

    return render(request, 'showroute.html', context) 

    #return render(request,'edges.html')    
    #return render(request, 'edges.html', context)
    #return render(request, 'showroute.html', context) 

    
    


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

#Functionality from time library
import time

#Pycuda
#from pycuda.compiler import SourceModule
import pycuda.compiler as SourceModule

#to run CUDA in GPU using numbapro
from numba import cuda, vectorize, guvectorize
from numba import void, uint8, uint32, uint64, int32, int64, float32, float64, f8

#to mathematical calcs
import numpy as np 

def testGPU1(request):
    start = time.perf_counter()
    do_something()
    do_something()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} second(s)')

    mod = SourceModule("""
        void __global__ kernel_add_arrays(float* a, float* b, float* c, int length){
            int gid = threadIdx.x + blockDim.x*blockIdx.x;

            while(gid <length){
                c[gid] = a[gid] + b[gid];
                gid += blockDim.x*gridDim.x;
            }
        }
        """)
    #fund = mod.get_function("kernel_add_arrays")   

    return render(request,'showmap.html')
    
def do_something():
    print('Sleeping 1 second...')
    time.sleep(1)
    print("Done sleeping...")


def testGPU2(request):
    
    gpu = cuda.get_current_device()

    #Check the number of grids, block and threads
    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))

    #To check if cuda recognize GPU
    #print(cuda.gpus)

    
    return render(request,'showmap.html')

def CPUClasses(request):
    
    #Timer to measure how it takes to finish
    start = time.perf_counter()
    initModelClasses()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} second(s)')


    return render(request,'showmap.html')

from route.models import SegmentStreet

def initModelClasses():
    
    
    #Constants and global values
    RHO_SEGMENT = .02

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
    MIN_GREEN_TIME_HORIZONTAL = 20

    global MAX_GREEN_TIME_HORIZONTAL        
    MAX_GREEN_TIME_HORIZONTAL = 21

    global MIN_GREEN_TIME_VERTICAL      
    MIN_GREEN_TIME_VERTICAL = 20

    global MAX_GREEN_TIME_VERTICAL      
    MAX_GREEN_TIME_VERTICAL = 21

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
    samples = int((TOTAL_TIME+1)*(1/SUB_STEP_TIME)  )

    #Values of Zmax in each segment
    ZMAX = np.zeros((9), dtype=float)
    for key in Segments:
        Segment = Segments[key]
        ZMAX[Segment.numberOfSegment-1]= Segment.limitOfCars

    carsDischarged = np.zeros(simulations, dtype=float) 

    counterSimulations = 0
    #simulations cycle
    for green_time_horizontal in range(MIN_GREEN_TIME_HORIZONTAL,MAX_GREEN_TIME_HORIZONTAL):
        for green_time_vertical in range(MIN_GREEN_TIME_VERTICAL,MAX_GREEN_TIME_VERTICAL):

            
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

            while(t<TOTAL_TIME): 
                #Samples each iteration
                tspan =numpy.linspace(t,t+STEP_TIME,int(STEP_TIME/SUB_STEP_TIME))

                #Update in integration value
                zp = odeint(updateCycle, z0, tspan,
                args=(Segments, alphas, aks, counter,
                currentState, time, weight52, 
                weight54, h15, h35, t, green_time_horizontal,
                green_time_vertical, weight52s, weight54s, 
                h15s, h35s, G, isPlotting, ZMAX))

                #Saving last value in array to use it in next cycle         
                z0 = zp[-1]                             

                #To save all values of zps
                zps = np.append(zps,zp, axis=0)

                counter+=1
                t +=STEP_TIME
            counterSimulations +=1
        
    if (isPlotting):
        plot(zps, h15s, h35s, weight54s, weight52s)



    
    

    
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
    #setting time to stateMachine
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
        #   pdb.set_trace()     
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
#       alpha = ((ALPHA_MAX-ALPHA_MIN)/(segmentS.rho-zd[segmentS.numberOfSegment-1]))*(zd[segmentS.numberOfSegment-1]-segmentS.limitOfCars)+ALPHA_MIN
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

        """plt.figure("Alpha, Ak")
        plt.subplot(2,1,1)
        plt.xlabel('Time')
        plt.ylabel("Alpha")
        plt.plot(alphas)

        plt.subplot(2,1,2)
        plt.xlabel('Time')
        plt.ylabel("Ak")
        plt.plot(aks)

        plt.figure("Blocks")

        plt.subplot(3,4,1)
        plt.xlabel('Time')
        plt.ylabel("z7")
        plt.plot(zCars[6])

        plt.subplot(3,4,2)
        plt.xlabel('Time')
        plt.ylabel("z3")
        plt.plot(zCars[2])

        plt.subplot(3,4,3)
        plt.xlabel('Time')
        plt.ylabel("z4")
        plt.plot(zCars[3])

        plt.subplot(3,4,4)
        plt.xlabel('Time')
        plt.ylabel("z9")
        plt.plot(zCars[8])

        plt.subplot(3,4,5)
        plt.xlabel('Time')
        plt.ylabel("z6")
        plt.plot(zCars[5])

        plt.subplot(3,4,6)
        plt.xlabel('Time')
        plt.ylabel("z1")
        plt.plot(zCars[0])

        plt.subplot(3,4,7)
        plt.xlabel('Time')
        plt.ylabel("z2")
        plt.plot(zCars[1])

        plt.subplot(3,4,8)
        plt.xlabel('Time')
        plt.ylabel("z8")
        plt.plot(zCars[7])

        plt.subplot(3,4,9)
        plt.xlabel('Time')
        plt.ylabel("z5")
        plt.plot(zCars[4])

        #Source-sink criteria
        plt.figure("Source-Sink criteria")
        plt.subplot(1,1,1)
        plt.xlabel('simulation')
        plt.ylabel("cars discharged")
        plt.plot(carsDischarged)


        plt.show()"""





#AQUÍ TERMINA EL RESPALDO DE VIEWS