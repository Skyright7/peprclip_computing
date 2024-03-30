from rest_framework import serializers
from generate_by_gaussian import models

class task_list_serializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.task_list
        fields = '__all__'

