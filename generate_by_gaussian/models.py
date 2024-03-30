from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

class task_list(models.Model):
    # 任务名称，检索用
    task_name = models.CharField(max_length=100)
    # 生成基底（种子）文件地址
    data_path = models.TextField()
    # 在候选集中随机选出几个多肽作为基底
    num_base_peps = models.IntegerField()
    # 每个sample为底生成几个候选多肽
    num_peps_per_base = models.IntegerField()
    # 生成的多肽的最短长度(筛种子的条件之一)
    min_length = models.IntegerField()
    # 生成的多肽的最长长度
    max_length = models.IntegerField()
    # 设置高斯噪音的方差范围（这个值越低，生成的跟原样本就越像，越高，越不像）
    # 方差下界
    sample_variances_down = models.IntegerField()
    # 方差上界
    sample_variances_up = models.IntegerField()
    # 跳采方差步长
    sample_variances_step = models.IntegerField()
    # 最终输出存放地址
    output_path = models.TextField()
    #用户注释
    user_comment = models.TextField(default='NA')
    #创建时间
    created = models.DateTimeField(default=timezone.now)
    # #更新时间
    # updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.task_name

    class Meta:
        ordering = ['-created']