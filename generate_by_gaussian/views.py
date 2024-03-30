from django.shortcuts import render

# Create your views here.
from generate_by_gaussian.models import task_list
from generate_by_gaussian.serializers import task_list_serializer

from rest_framework import filters

from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from generate_by_gaussian.generate_script import generate_script

class task_List_viewset(viewsets.ModelViewSet):
    queryset = task_list.objects.all()
    serializer_class = task_list_serializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['task_name']

    @action(detail=True)
    def do_generate_gaussian(self,request,pk=None):
        current_task_list = task_list.objects.get(pk=pk)
        task_name = current_task_list.task_name
        data_path = current_task_list.data_path
        num_base_peps = current_task_list.num_base_peps
        num_peps_per_base = current_task_list.num_peps_per_base
        min_length = current_task_list.min_length
        max_length = current_task_list.max_length
        sample_variances_down = current_task_list.sample_variances_down
        sample_variances_up = current_task_list.sample_variances_up
        sample_variances_step = current_task_list.sample_variances_step
        output_path = current_task_list.output_path
        out_file_path = generate_script(task_name,data_path,num_base_peps,num_peps_per_base,min_length,max_length,sample_variances_down,sample_variances_up,sample_variances_step,output_path)
        return Response({'status': f'you mission is successful updated,and the file save path is:{out_file_path}'})


