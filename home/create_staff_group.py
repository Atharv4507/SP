from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission


class Command(BaseCommand):
    help = 'Create Staff group and assign permissions'

    def handle(self, *args, **options):
        # Create or retrieve a user group (replace 'Staff' with your group name)
        staff_group, created = Group.objects.get_or_create(name='Staff')

        # Get the permissions for your models (replace with the correct permission names)
        view_permission_result = Permission.objects.get(codename='view_prediction result')
        view_permission_result2 = Permission.objects.get(codename='view_prediction result2')
        view_permission_result3 = Permission.objects.get(codename='view_prediction result3')

        # Assign the permissions to the user group
        staff_group.permissions.add(view_permission_result, view_permission_result2, view_permission_result3)

        self.stdout.write(self.style.SUCCESS('Successfully created Staff group and assigned permissions.'))
