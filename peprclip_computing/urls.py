"""peprclip_computing URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from django.urls import include

from rest_framework.routers import DefaultRouter
from generate_by_gaussian.views import task_List_viewset
from filter_by_clip.views import Clip_List_viewset
from clipFilter.views import ClipViewset
from gaussianGenerate.views import gaussianTaskListViewset

router = DefaultRouter()
router.register(r'generate_by_gaussian', task_List_viewset)
router.register(r'filter_by_clip', Clip_List_viewset)
router.register(r'clipFilter', ClipViewset)
router.register(r'gaussianGenerate', gaussianTaskListViewset)

urlpatterns = [
    path('api/', include(router.urls)),
]
