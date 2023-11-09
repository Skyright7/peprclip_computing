from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone


# Create your models here.
# 当前假设模型服务是通过Django包裹做成服务，然后在docker上暴露一个调用接口的微服务架构，那么当前主业务的表单应该是一个功能列表。
class targetList(models.Model):
    #受体名称
    target_name = models.CharField(max_length=100)
    #受体序列
    target_seq = models.TextField()
    #生成序列json
    output_json = models.JSONField(null=True)
    #num_per_target
    num_per_target = models.IntegerField()
    #用户注释
    user_comment = models.TextField()
    #创建时间
    created = models.DateTimeField(default=timezone.now)
    #更新时间
    updated = models.DateTimeField(auto_now=True)
    def __str__(self):
        return self.target_name

    class Meta:
        ordering = ['-created']
