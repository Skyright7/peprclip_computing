from django.db import models

class gaussianGenerateTaskList(models.Model):
    # 任务名称，检索用
    taskName = models.CharField(max_length=100)
    # 生成基底（种子）文件地址
    dataPath = models.TextField()
    # 在候选集中随机选出几个多肽作为基底
    numBasePeps = models.IntegerField()
    # 每个sample为底生成几个候选多肽
    numPepsPerBase = models.IntegerField()
    # 生成的多肽的最短长度(筛种子的条件之一)
    minLength = models.IntegerField()
    # 生成的多肽的最长长度(筛种子的条件之一)
    maxLength = models.IntegerField()
    # 设置高斯噪音的方差范围（这个值越低，生成的跟原样本就越像，越高，越不像）
    # 方差下界
    sampleVariancesDown = models.IntegerField()
    # 方差上界
    sampleVariancesUp = models.IntegerField()
    # 跳采方差步长
    sampleVariancesStep = models.IntegerField()

    def __str__(self):
        return self.taskName

    class Meta:
        ordering = ['-id']