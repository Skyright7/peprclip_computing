from rest_framework import serializers
from gaussianGenerate import models

class gaussianTaskListSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.gaussianGenerateTaskList
        fields = '__all__'

