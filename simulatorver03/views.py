from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse

#utilities
import xml.dom.minidom 

#from qiskit import Aer


def parse(request):
    #parse the information of the road
    #xmldoc = xml.dom.minidom.parse('mapSimpleMex.osm')
    xmldoc = xml.dom.minidom.parse('mapComplexMex.osm')

    nodes = xmldoc.getElementsByTagName('node')
    ways = xmldoc.getElementsByTagName('way')

    #for node in nodes:
        #print(node)
    
    message = str(len(nodes))+' nodes,'+ str(len(ways))+ ' ways' 
    
    #xmldoc2 = xml.dom.minidom.parseString(  )
    """readbitlist = xmldoc.getElementsByTagName('node')
    values = []
    for s in readbitlist :
        x = s.attributes['node'].value
        values.append(x)
    return render(request, 'parse.html', {'values': values})"""



    return HttpResponse(message)