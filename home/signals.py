from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User, Permission
from django.contrib.contenttypes.models import ContentType
from home.models import PredictionResult, PredictionResult2, PredictionResult3

@receiver(post_save, sender=User)
def set_staff_status(sender, instance, created, **kwargs):
    if created:
        instance.is_staff = True
        instance.save()


@receiver(post_save, sender=User)
def grant_model_permissions(sender, instance, created, **kwargs):
    if created:
        # Define the models for which you want to grant permission
        models = [PredictionResult, PredictionResult2, PredictionResult3]

        for model in models:
            app_label = model._meta.app_label
            codename = f"view_{model._meta.model_name}"

            # Create or get the permission
            permission, created = Permission.objects.get_or_create(
                codename=codename,
                content_type=ContentType.objects.get_for_model(model),
                name=f"Can view {model._meta.verbose_name}",
            )

            # Add the permission to the user
            instance.user_permissions.add(permission)
