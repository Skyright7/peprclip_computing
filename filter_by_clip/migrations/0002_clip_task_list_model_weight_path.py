# Generated by Django 4.1 on 2024-03-30 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('filter_by_clip', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='clip_task_list',
            name='model_weight_path',
            field=models.TextField(default='./model_weight/clip/canonical_pepprclip_4-22-23.ckpt'),
        ),
    ]
