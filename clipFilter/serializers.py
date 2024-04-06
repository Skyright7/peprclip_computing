from rest_framework import serializers
from clipFilter import models

class clipListSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.clipTaskList
        fields = '__all__'

