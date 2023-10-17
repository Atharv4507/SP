from datetime import date, timedelta

from django.contrib import admin
from django.contrib.auth.decorators import login_required

from .models import SalesInput, Prediction, PredictionResult, pcw, PredictionResult2, PredictionResult3
# Register your models here.
admin.site.register(SalesInput)
admin.site.register(Prediction)
# admin.site.register(PredictionResult)
# admin.site.register(PredictionResult2)
# admin.site.register(PredictionResult3)
admin.site.register(pcw)


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.has_perm("home.view_prediction result"):
            # Filter the queryset based on your criteria for staff users
            seven_days_ago = date.today() - timedelta(days=7)
            qs = qs.filter(input_date__gte=seven_days_ago)
        return qs


@admin.register(PredictionResult2)
class PredictionResultAdmin(admin.ModelAdmin):
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.has_perm("home.view_prediction result2"):
            # Filter the queryset based on your criteria for staff users
            seven_days_ago = date.today() - timedelta(days=7)
            qs = qs.filter(input_date__gte=seven_days_ago)
        return qs


@admin.register(PredictionResult3)
class PredictionResultAdmin(admin.ModelAdmin):
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.has_perm("home.view_prediction result3"):
            # Filter the queryset based on your criteria for staff users
            seven_days_ago = date.today() - timedelta(days=7)
            qs = qs.filter(input_date__gte=seven_days_ago)
        return qs
