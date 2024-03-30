from django.db import models

# Create your models here.
from django.utils import timezone

class clip_task_list(models.Model):
    # 任务名称，检索用
    task_name = models.CharField(max_length=100)
    # 筛选选择的权重文件的地址
    model_weight_path = models.TextField(default='./model_weight/clip/canonical_pepprclip_4-22-23.ckpt')
    # 要筛选的多肽文件的地址
    peptides_path = models.TextField()
    # 要筛选的受体蛋白的序列
    target_seq = models.TextField()
    # 筛选出TOP几作为最终输出
    peps_per_target = models.IntegerField()
    # 受体蛋白名称
    target_name = models.CharField(max_length=100)
    # 最终输出存放地址
    output_base_path = models.TextField()
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