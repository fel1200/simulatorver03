from django.db import models

#Arrayfields to storage array in the model
from django.contrib.postgres.fields import ArrayField


# Create your models here.
class SegmentStreet(models.Model):
	#init Method
	def __init__(self, name, numberOfSegment, type_of_segment, orientation, initialCars,
		currentCars, tempCurrentCars, limitOfCars, inputs, orientation_inputs,
		outputs, orientation_outputs, rho, alpha, ak, z, zp):
		self.name = name
		self.numberOfSegment = numberOfSegment
		self.type_of_segment = type_of_segment
		self.orientation = orientation
		self.initialCars = initialCars
		self.currentCars = currentCars
		self.tempCurrentCars = tempCurrentCars
		self.limitOfCars = limitOfCars
		#self.alphaValue = alphaValue
		self.inputs = inputs
		self.orientation_inputs = orientation_inputs
		self.outputs = outputs
		self.orientation_outputs = orientation_outputs
		#self.ak_outputs = ak_outputs
		self.rho = rho
		self.alpha = alpha
		self.ak = ak
		self.z = z 
		self.zp = zp
	#Number of segment
	numberOfSegment = models.IntegerField()

	#Name
	name = models.CharField(max_length = 50)


	TYPE_OF_SEGMENTS =[
	('ST','Street'),
	('IN','Intersection'),
	('IT','Traffic light intersection'),
	('SO','Source'),
	('SI','Sink'),

	]

	type_of_segment = models.CharField(
		max_length = 2,
		choices = TYPE_OF_SEGMENTS,
		default ='ST')

	#We only use horizontal and vertical at the moment
	#In the next steps include a different method
	ORIENTATIONS =[
	('HO','Horizontal'),
	('VE','Vertical'),
	('IN','Intersection'),
	]

	orientation = models.CharField(
		max_length = 2,
		choices = ORIENTATIONS,
		default = "HO"
		)
	initialCars = models.IntegerField()
	currentCars = models.IntegerField()
	tempCurrentCars = models.IntegerField()
	limitOfCars = models.IntegerField()
	#alphaValue =models.FloatField()
	
	#Entry and exit street segments
	inputs = ArrayField(models.CharField(max_length = 2))
	outputs = ArrayField(models.CharField(max_length = 2))


	#Input and ouput orientation to know block's direction
	orientation_inputs = ArrayField(models.CharField(max_length = 2))
	orientation_outputs = ArrayField(models.CharField(max_length = 2))


	rho = models.FloatField()
	alpha = models.FloatField()
	ak = models.FloatField()
	
	z = models.FloatField()
	zp = models.FloatField()

	def __str__(self):
		return 'name: {}, type: {}, currentCars: {}, limitOfCars: {}, inputs: {}, orientation_inputs: {}, outputs: {}, orientation_outputs: {}'.format(self.name, self.type_of_segment, self.currentCars, self.limitOfCars, self.inputs, self.orientation_inputs, self.outputs, self.orientation_outputs )

