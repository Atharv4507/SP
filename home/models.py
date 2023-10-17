from django.contrib.auth.decorators import login_required
from django.db import models


# Create your models here.
class SalesInput(models.Model):
    feature1 = models.FloatField()
    feature2 = models.FloatField()

    # Add more fields as needed

    def __str__(self):
        return f"Sales Input - {self.pk}"


class Prediction(models.Model):
    date = models.DateField()

    def __str__(self):
        return str(self.date)


class pcw(models.Model):
    CATEGORY_CHOICES = [
        ('Health and beauty', 'Health and Beauty'),
        ('Electronic accessories', 'Electronic Accessories'),
        ('Home and lifestyle', 'Home and Lifestyle'),
        ('Fashion accessories', 'Fashion accessories'),
        ('Food and beverages', 'Food and beverages'),
        ('Sports and travel', 'Sports and travel'),
    ]

    input_category = models.CharField(
        max_length=50, choices=CATEGORY_CHOICES)
    input_date = models.DateField()

    def __str__(self):
        return str(self.input_category)


class PredictionResult(models.Model):
    input_date = models.DateField()
    demand_prediction = models.FloatField()
    supply_prediction = models.FloatField()

    def __str__(self):
        return str(self.input_date)

    class Meta:
        permissions = [
            ("view_prediction result", "Can view prediction result"),
            ("view_prediction result2", "Can view prediction result2"),
            ("view_prediction result3", "Can view prediction result3"),
        ]


class PredictionResult2(models.Model):
    CATEGORY_CHOICES = [
        ('Health and beauty', 'Health and Beauty'),
        ('Electronic accessories', 'Electronic Accessories'),
        ('Home and lifestyle', 'Home and Lifestyle'),
        ('Fashion accessories', 'Fashion accessories'),
        ('Food and beverages', 'Food and beverages'),
        ('Sports and travel', 'Sports and travel'),
    ]

    input_category = models.CharField(
        max_length=50, choices=CATEGORY_CHOICES)
    input_date = models.DateField()
    demand_prediction = models.FloatField()
    supply_prediction = models.FloatField()
    Demand_Graph = models.ImageField(upload_to='static/images/', default="static/images/Demand_plot.jpg")
    Supply_Graph = models.ImageField(upload_to='static/images/', default="static/images/Supply_plot.jpg")
    Demand_And_Supply_Graph = models.ImageField(upload_to='static/images/',
                                                default="static/images/DemandAndSupply_plot.jpg")
    Dtree_Scatter_Graph = models.ImageField(upload_to='static/images/',
                                            default="static/images/dtree_scatter_plot.png")

    def __str__(self):
        return str(self.input_category)


class PredictionResult3(models.Model):

    CITY_CHOICES = [
        ('Bangalore', 'Bangalore'),
        ('Chennai', 'Chennai'),
        ('Delhi', 'Delhi'),
        ('Hyderabad', 'Hyderabad'),
        ('Kanpur', 'Kanpur'),
        ('Kolkata', 'Kolkata'),
        ('Mumbai', 'Mumbai'),
        ('Pune', 'Pune'),
        ('Surat', 'Surat'),
    ]
    CATEGORY_CHOICES = [
        ('Health and beauty', 'Health and Beauty'),
        ('Electronic accessories', 'Electronic Accessories'),
        ('Home and lifestyle', 'Home and Lifestyle'),
        ('Fashion accessories', 'Fashion accessories'),
        ('Food and beverages', 'Food and beverages'),
        ('Sports and travel', 'Sports and travel'),
    ]

    input_city = models.CharField(max_length=50, choices=CITY_CHOICES)
    input_category = models.CharField(
        max_length=50, choices=CATEGORY_CHOICES)
    input_date = models.DateField()
    demand_prediction = models.FloatField()
    supply_prediction = models.FloatField()
    Demand_Graph = models.ImageField(upload_to='static/images/', default="static/images/Demand_plot.jpg")
    Supply_Graph = models.ImageField(upload_to='static/images/', default="static/images/Supply_plot.jpg")
    Demand_And_Supply_Graph = models.ImageField(upload_to='static/images/',
                                                default="static/images/DemandAndSupply_plot.jpg")
    Dtree_Scatter_Graph = models.ImageField(upload_to='static/images/',
                                            default="static/images/dtree_scatter_plot.png")

    def __str__(self):
        return str(self.input_city)


