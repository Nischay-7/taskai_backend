from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TaskViewSet, suggest_subtasks, register
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

router = DefaultRouter()
router.register(r'tasks', TaskViewSet, basename='task')


urlpatterns = [
    path('', include(router.urls)),
    path('register/', register),
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('ai/suggest/', suggest_subtasks),
]
