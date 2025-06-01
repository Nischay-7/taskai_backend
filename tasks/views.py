import requests
import os, re
from rest_framework import viewsets, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth.models import User
from .models import Task
from .serializers import TaskSerializer, UserSerializer
from django.conf import settings
from openai import OpenAI
from huggingface_hub import InferenceClient
from rest_framework.permissions import IsAuthenticated
from rest_framework import status as http_status

# Task CRUD
class TaskViewSet(viewsets.ModelViewSet):
    serializer_class = TaskSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Task.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()

        # Toggle status logic
        current_status = instance.status.lower()
        new_status = "done" if current_status == "pending" else "pending"
        request.data["status"] = new_status  # Overwrite status in request

        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        is_done = request.data.get("isDone", None)
        if is_done is not None:
            request.data["status"] = "done" if is_done else "pending"

        return Response(serializer.data, status=http_status.HTTP_200_OK)

# Register API
@api_view(['POST'])
def register(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({'message': 'User registered successfully'})
    return Response(serializer.errors, status=400)

# AI suggest endpoint
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def suggest_subtasks(request):
    task_id = request.data.get("task_id")
    if not task_id:
        return Response({"error": "task_id field is required"}, status=400)

    try:
        task = Task.objects.get(id=task_id, user=request.user)
    except Task.DoesNotExist:
        return Response({"error": "Task not found or you do not have permission to access it."}, status=404)

    task_content = task.description or task.title
    if not task_content:
        return Response({"error": "Task has no content (title or description) to suggest subtasks for."}, status=400)

    prompt = f"""Break this task into 5 subtasks and suggest improvements if needed.

Task: {task_content}

Output format:
1. Subtask Title
- Improvement: ..."""

    try:
        client = InferenceClient(model="google/flan-t5-large")

        generated_text = client.text_generation(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            stop=["\n\n"]  # correct usage for flan-t5-xxl
        )

        message = generated_text.strip()

        # Parse the result
        subtasks = []
        items = re.split(r'\n?\s*\d+\.\s+', message)
        items = [item.strip() for item in items if item.strip()]

        for item in items[:5]:
            parts = re.split(r'\n\s*-\s*Improvement:\s*', item)
            title = parts[0].strip()
            improvements = [imp.strip() for imp in parts[1:]] if len(parts) > 1 else []
            subtasks.append({
                "title": title,
                "improvements": improvements
            })

        return Response({"suggestion": subtasks})

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debugging support
        return Response({"error": f"AI generation failed: {str(e)}"}, status=500)