# Generated by Django 4.2.5 on 2023-09-29 02:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0003_predictionresult'),
    ]

    operations = [
        migrations.RenameField(
            model_name='predictionresult',
            old_name='demand',
            new_name='demand_prediction',
        ),
        migrations.RenameField(
            model_name='predictionresult',
            old_name='date',
            new_name='input_date',
        ),
        migrations.RenameField(
            model_name='predictionresult',
            old_name='supply',
            new_name='supply_prediction',
        ),
    ]