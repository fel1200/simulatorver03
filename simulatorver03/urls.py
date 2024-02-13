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

#Home view
from route import a_homePage as home
##Route views
from route import a_generalMap 
##CPU classes
from route import CPUClasses
##CPU arreglos
from route import CPUArrays

#GPU simulator, integration CPU
from route import GPU_NI
from route import GPU_NIRespa7Dic2023
from route import GPU_testGama
from route import GPU_integrate

#General
urlpatterns = [
    #GENERAL
    #HOME
    path('', home.home, name ='home'),
    
    #Parse info
    path('parse/', a_generalMap.parse), # Only to get info from nodes and ways in a vehicular network

    #General maps
    path('showmapRandom',a_generalMap.showmapRandom,name='showmapRandom'), #Muestra un mapa random
    path('<str:lat1>,<str:long1>,<str:lat2>,<str:long2>',
        a_generalMap.showRoute,name='showroute'), #Muestra una ruta del mapa
    path('showmapMexicoCityPoints/',
     a_generalMap.showmapMexicoCityPoints), # Pinta la ruta de un punto A a un punto B del Centro histórico de la Ciudad de
    path('showmapSelector/<int:selector>', a_generalMap.showmapSelector),

    #ADMIN
    path('admin/', admin.site.urls), #admin to check models 
    #END OF GENERAL
]
#CPU
urlpatterns += [
    #CPU
    #Classes
    path('CPUClasses/', CPUClasses.CPUClasses),
    #Arrays
    path('CPUArrays/', CPUArrays.CPUArrays),
    ]
#GPU
urlpatterns += [
   #GPU
    #without integration method respa
    path('GPU_NIRespa7Dic2023/', GPU_NIRespa7Dic2023.GPU_NIRespa7Dic2023),
    #without integration method
    path('GPU_NI/', GPU_NI.GPU_NI),
    #test of generic 
    path('GPU_testGeneric/', GPU_testGama.GPU_testGeneric, name="GPU_testGeneric"), #some generic tests
    #test of integration generic Dr. Gamaliel working on GPU
    path('GPU_testGama/', GPU_testGama.GPU_testGama, name="GPU_testGama"), #Generic GPU integration code Dr. Gamaliel
    #with integration method
    path('GPU_integrate/', GPU_integrate.GPU_integrate), #GPU Integration to the model
]