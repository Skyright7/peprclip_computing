from django.shortcuts import render

# Create your views here.

# Create your views here.
from filter_by_clip.models import clip_task_list
from filter_by_clip.serializers import clip_list_serializer

from rest_framework import filters

from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from filter_by_clip.clip_script import do_clip

class Clip_List_viewset(viewsets.ModelViewSet):
    queryset = clip_task_list.objects.all()
    serializer_class = clip_list_serializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['task_name']

    @action(detail=True)
    def do_generate_gaussian(self,request,pk=None):
        current_task_list = clip_task_list.objects.get(pk=pk)
        task_name = current_task_list.task_name
        peptides_path = current_task_list.peptides_path
        peps_per_target = current_task_list.peps_per_target
        target_seq = current_task_list.target_seq
        target_name = current_task_list.target_name
        output_base_path = current_task_list.output_base_path
        model_weight_path = current_task_list.model_weight_path
        # out_file_path = generate_script(task_name,data_path,num_base_peps,num_peps_per_base,min_length,max_length,sample_variances_down,sample_variances_up,sample_variances_step,output_path)
        out_file_path = do_clip(target_seq,target_name, peptides_path,peps_per_target,output_base_path,task_name,model_weight_path)
        return Response({'status': f'you mission is successful updated,and the file save path is:{out_file_path}'})


