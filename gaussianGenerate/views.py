from django.shortcuts import render

# Create your views here.
from gaussianGenerate.models import gaussianGenerateTaskList
from gaussianGenerate.serializers import gaussianTaskListSerializer

from rest_framework import filters

from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from gaussianGenerate.generateScript import generate_script

class gaussianTaskListViewset(viewsets.ModelViewSet):
    queryset = gaussianGenerateTaskList.objects.all()
    serializer_class = gaussianTaskListSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['taskName']

    @action(detail=True)
    def doGenerate(self,request,pk=None):
        current_task_list = gaussianGenerateTaskList.objects.get(pk=pk)
        task_name = current_task_list.taskName
        data_path = current_task_list.dataPath
        num_base_peps = current_task_list.numBasePeps
        num_peps_per_base = current_task_list.numPepsPerBase
        min_length = current_task_list.minLength
        max_length = current_task_list.maxLength
        sample_variances_down = current_task_list.sampleVariancesDown
        sample_variances_up = current_task_list.sampleVariancesUp
        sample_variances_step = current_task_list.sampleVariancesStep
        out_file_path = generate_script(task_name,data_path,num_base_peps,num_peps_per_base,min_length,max_length,sample_variances_down,sample_variances_up,sample_variances_step)
        base_dir = '/home/app/lgy'
        relative_path = out_file_path
        # 去掉开头的"./"
        if out_file_path.startswith('./'):
            relative_path = out_file_path[2:]
        gaussianOutPath = base_dir + '/' + relative_path
        return Response(gaussianOutPath,status=status.HTTP_200_OK)


