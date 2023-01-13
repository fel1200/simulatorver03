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

from simulatorver03 import views as local_views

#Map functionality
#from route.views import showroute, showmap, showmapSimulator, showmapInitial, testGPU1, testGPU2

from route import views as route_views
from route import viewsPythonArray as pythonArray_views
from route import viewsGPU as GPU_views
from route import testIntegration as testIntegration_views
from route import viewsGPUIntegrateGPU as integrateGPU

urlpatterns = [
    path('admin/', admin.site.urls),
    path('simulator03/', local_views.parse),
    path('<str:lat1>,<str:long1>,<str:lat2>,<str:long2>',route_views.showroute,name='showroute'),
    path('',route_views.showmap,name='showmap'),
    path('showmapSimulator/', route_views.showmapSimulator),
    path('showmapInitial/', route_views.showmapInitial),
    path('testGPU1/', route_views.testGPU1),
    path('testGPU2/', route_views.testGPU2),
    path('CPUClasses/', route_views.CPUClasses),
    path('CPUArray/', pythonArray_views.pythonArray),
    path('GPU/', GPU_views.pythonGPU),
    #path('IntegrateGPU/', GPU_views.pythonGPUIntegrate),
    path('TestIntegrationGPU/', testIntegration_views.testIntegrationGPU), #Generic GPU integration code 
    path('IntegrateGPU/', integrateGPU.pythonGPUIntegrate), #GPU Integration to the model

]
