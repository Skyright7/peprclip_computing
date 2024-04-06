from django.shortcuts import render

# Create your views here.

# Create your views here.
from clipFilter.models import clipTaskList
from clipFilter.serializers import clipListSerializer

from rest_framework import filters

from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from clipFilter.clipScript import do_clip
from pathlib import Path

class ClipViewset(viewsets.ModelViewSet):
    queryset = clipTaskList.objects.all()
    serializer_class = clipListSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['taskName']

    @action(detail=True)
    def do_generate_gaussian(self,request,pk=None):
        current_task_list = clipTaskList.objects.get(pk=pk)
        taskName = current_task_list.taskName
        peptidesPath = current_task_list.peptidesPath
        pepsPerTarget = current_task_list.pepsPerTarget
        targetSeq = current_task_list.targetSeq
        targetName = current_task_list.targetName
        out_file_path = do_clip(targetSeq,targetName, peptidesPath,pepsPerTarget,taskName)
        base_dir = '/home/app/lgy'
        relative_path = out_file_path
        # 去掉开头的"./"
        if out_file_path.startswith('./'):
            relative_path = out_file_path[2:]
        clipOutPath = base_dir + '/' + relative_path
        return Response(clipOutPath,status=status.HTTP_200_OK)


