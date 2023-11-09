"""simulatorver03 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

##Route views
from route import generalMap 
from route import viewsPythonArray as pythonArray_views
from route import viewsGPU as GPU_views
from route import testIntegration as testIntegration_views
from route import viewsGPUIntegrateGPU as integrateGPU
from route import homePage as home

urlpatterns = [

    #General maps
    path('',generalMap.showmap,name='showmap'), #Muestra un mapa random
    path('<str:lat1>,<str:long1>,<str:lat2>,<str:long2>',
        generalMap.showroute,name='showroute'), #Muestra una ruta del punto A al B
    path('showmapSimulator/',
     generalMap.showmapSimulator), # Pinta la ruta de un punto A a un punto B del Centro histórico de la Ciudad de
    path('showmapInitial/<int:selector>', generalMap.showmapInitial),
    
    path('testGPU1/', generalMap.testGPU1),
    path('testGPU2/', generalMap.testGPU2),
    path('CPUClasses/', generalMap.CPUClasses),
    path('CPUArray/', pythonArray_views.pythonArray),
    path('GPU/', GPU_views.pythonGPU),
    #path('IntegrateGPU/', GPU_views.pythonGPUIntegrate),
    path('TestIntegrationGPU/', testIntegration_views.testIntegrationGPU, name="gpuGenericIntegration"), #Generic GPU integration code 
    path('IntegrateGPU/', integrateGPU.pythonGPUIntegrate), #GPU Integration to the model

    ##Admin
    path('admin/', admin.site.urls), #admin to check models 
    path('simulator03/', generalMap.parse), # Only to get info from nodes and ways in a vehicular network

]

#To add html views

urlpatterns += [
    path('home/', home.home, name ='home')
]
