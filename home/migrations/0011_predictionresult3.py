# Generated by Django 4.2.5 on 2023-10-14 16:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0010_rename_input_image_predictionresult2_demand_graph_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionResult3',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_city', models.CharField(choices=[('Bangalore', 'Bangalore'), ('Chennai', 'Chennai'), ('Delhi', 'Delhi'), ('Hyderabad', 'Hyderabad'), ('Kanpur', 'Kanpur'), ('Kolkata', 'Kolkata'), ('Mumbai', 'Mumbai'), ('Pune', 'Pune'), ('Surat', 'Surat')], max_length=50)),
                ('input_category', models.CharField(choices=[('Health and beauty', 'Health and Beauty'), ('Electronic accessories', 'Electronic Accessories'), ('Home and lifestyle', 'Home and Lifestyle'), ('Fashion accessories', 'Fashion accessories'), ('Food and beverages', 'Food and beverages'), ('Sports and travel', 'Sports and travel')], max_length=50)),
                ('input_date', models.DateField()),
                ('demand_prediction', models.FloatField()),
                ('supply_prediction', models.FloatField()),
                ('Demand_Graph', models.ImageField(default='static/images/Demand_plot.jpg', upload_to='static/images/')),
                ('Supply_Graph', models.ImageField(default='static/images/Supply_plot.jpg', upload_to='static/images/')),
                ('Demand_And_Supply_Graph', models.ImageField(default='static/images/DemandAndSupply_plot.jpg', upload_to='static/images/')),
                ('Dtree_Scatter_Graph', models.ImageField(default='static/images/dtree_scatter_plot.png', upload_to='static/images/')),
            ],
        ),
    ]