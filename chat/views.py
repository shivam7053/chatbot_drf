from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .models import ChatMessage
from .serializers import ChatMessageSerializer


# Load model once globally (so it doesn’t reload on every request)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

class ChatBotAPIView(APIView):
    def post(self, request):
        user_message = request.data.get('message', '')

        # Encode user message
        input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')

        # Generate response
        chat_history_ids = model.generate(
            input_ids,
            max_length=80,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            top_p=0.92,
            top_k=50,
            temperature=0.7,
        )

        bot_reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Save in DB
        chat = ChatMessage.objects.create(user_message=user_message, bot_reply=bot_reply)
        serializer = ChatMessageSerializer(chat)
        return Response(serializer.data, status=status.HTTP_200_OK)
