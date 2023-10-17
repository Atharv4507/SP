from django.urls import path
from home import views

urlpatterns = [
    path('sign', views.sign, name='sign'),
    path('', views.loginp, name='loginp'),
    path('logoutp', views.logoutp, name='logoutp'),
    path('base', views.base, name='base'),
    path('mainhome', views.mainhome, name='home'),
    # path('accounts/login/', views.predict_demand_supply_dtree, name='predict'),
    path('predict/', views.predict_demand_supply_dtree, name='predict'),
    path('prediction_results', views.predict_demand_supply_dtree, name='predictResult'),
    path('pcw', views.pcw, name='pcw'),
    path('prediction_results2', views.pcw, name='predictResult2'),
    path('pcacw', views.pcacw, name='pcacw'),
    path('prediction_results3', views.pcacw, name='predictResult3')
]
