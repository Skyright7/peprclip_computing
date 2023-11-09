


# article/serializers.py

from rest_framework import serializers
from model_computing import models

class targetList_serializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.targetList
        fields = '__all__'

