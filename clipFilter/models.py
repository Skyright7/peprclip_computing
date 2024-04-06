from django.db import models

class clipTaskList(models.Model):
    # 任务名称，检索用
    taskName = models.CharField(max_length=100)
    # 要筛选的多肽文件的地址
    peptidesPath = models.TextField()
    # 要筛选的受体蛋白的序列
    targetSeq = models.TextField()
    # 筛选出TOP几作为最终输出
    pepsPerTarget = models.IntegerField()
    # 受体蛋白名称
    targetName = models.CharField(max_length=100)

    def __str__(self):
        return self.taskName

    class Meta:
        ordering = ['-id']