from rest_framework import serializers
from filter_by_clip import models

class clip_list_serializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.clip_task_list
        fields = '__all__'

