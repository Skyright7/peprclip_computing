from django.shortcuts import render

# Create your views here.
from model_computing.models import targetList
from model_computing.target_list_serializer import targetList_serializer

from rest_framework import viewsets

from rest_framework import filters

from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from model_computing.computing_script import out_computing_script

class targetList_viewset(viewsets.ModelViewSet):
    queryset = targetList.objects.all()
    serializer_class = targetList_serializer
    # permission_classes = [IsAdminUserOrReadOnly]
    # def perform_create(self, serializer):
    #     serializer.save(author=self.request.user)
    filter_backends = [filters.SearchFilter]
    search_fields = ['target_name']
    # @action(detail=True, methods=['post'])
    # def do_computing(self,request,pk=None):
    #     target = self.get_object()
    #     serializer = targetList_serializer(data=request.data)
    #     if serializer.is_valid():
    #         target_name = serializer.validated_data['target_name']
    #         target_seq = serializer.validated_data['target_seq']
    #         num_per_target = serializer.validated_data['num_per_target']
    #         out_json = out_computing_script(target_name,target_seq, './100k_denovo_for_FcRn.pkl', './pepprclip_2023-10-12.ckpt',num_per_target)
    #         target.set_target_name(target_name)
    #         target.set_target_seq(target_seq)
    #         target.set_num_per_target(num_per_target)
    #         target.set_output_json(out_json)
    #         target.save()
    #         return Response({'status': 'out_json set'})
    #     else:
    #         return Response(serializer.errors,
    #                         status=status.HTTP_400_BAD_REQUEST)
    @action(detail=True)
    def do_computing(self,request,pk=None):
        target = targetList.objects.get(pk=pk)
        target_name = target.target_name
        target_seq = target.target_seq
        num_per_target = target.num_per_target
        out_json = out_computing_script(target_name, target_seq, './100k_denovo_for_FcRn.pkl',
                                        './model_weight/clip/canonical_pepprclip_4-22-23.ckpt', num_per_target)
        target.output_json = out_json
        target.save()
        return Response({'status': 'out_json updated'})
