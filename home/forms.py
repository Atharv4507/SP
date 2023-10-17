from django import forms


class PredictionForm(forms.Form):
    date = forms.DateField(label='Date', widget=forms.DateInput(attrs={'type': 'date'}))
    # demand = forms.FloatField()
    # supply = forms.FloatField()

    # def __str__(self):
    #     return f'Date: {self.date}, Demand: {self.demand}, Supply: {self.supply}'


class PredictionForm_dtree(forms.Form):
    date = forms.DateField(label='Date', widget=forms.DateInput(attrs={'type': 'date'}))
