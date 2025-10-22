from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import spacy
from .models import ChatMessage
from .serializers import ChatMessageSerializer

# ✅ Load SpaCy model with word vectors (ensure installed in Dockerfile)
nlp = spacy.load("en_core_web_md")

# ✅ Precompute question embeddings once at startup
SPACE_QA = {
    "what is the milky way": "The Milky Way is the galaxy that contains our Solar System.",
    "what is a black hole": "A black hole is a region in space where gravity is so strong that not even light can escape.",
    "what is the sun": "The Sun is the star at the center of our Solar System.",
    "what is the moon": "The Moon is Earth's only natural satellite.",
    "what is nasa": "NASA is the National Aeronautics and Space Administration of the United States.",
    "what is the speed of light": "The speed of light is approximately 299,792 kilometers per second.",
    "what is the largest planet": "Jupiter is the largest planet in our Solar System.",
    "what is the smallest planet": "Mercury is the smallest planet in our Solar System.",
    "what is a comet": "A comet is an icy, small Solar System body that releases gases when near the Sun.",
    "what is an asteroid": "An asteroid is a small rocky body orbiting the Sun.",
    "what is a galaxy": "A galaxy is a massive system of stars, gas, dust, and dark matter bound together by gravity.",
    "what is a star": "A star is a massive, luminous sphere of plasma held together by its own gravity.",
    "what is space": "Space is the vast region beyond Earth's atmosphere.",
    "how many planets are there": "There are eight planets in our Solar System.",
    "what is mars": "Mars is the fourth planet from the Sun, known as the Red Planet.",
    "what is venus": "Venus is the second planet from the Sun and the hottest planet in our Solar System.",
    "what is the big bang": "The Big Bang theory describes how the universe began from an extremely hot, dense point.",
    "what is the international space station": "The ISS is a large spacecraft orbiting Earth, used for scientific research and living in space.",
    "what is gravity": "Gravity is a force that attracts two bodies toward each other.",
    "who was the first person in space": "Yuri Gagarin was the first human to travel into space in 1961."
}

# ✅ Precompute all vector docs once to speed up each request
QA_VECTORS = {q: nlp(q) for q in SPACE_QA.keys()}


class ChatBotAPIView(APIView):
    def post(self, request):
        user_message = request.data.get("message", "").strip().lower()

        # Handle empty message
        if not user_message:
            return Response(
                {"error": "Please provide a message."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user_doc = nlp(user_message)
        best_score = 0.0
        best_answer = "I'm not sure about that. Try asking something about space!"

        # ✅ Compare user message with precomputed vectors
        for question, question_doc in QA_VECTORS.items():
            similarity = user_doc.similarity(question_doc)
            if similarity > best_score:
                best_score = similarity
                best_answer = SPACE_QA[question]

        # ✅ Confidence threshold to avoid random matches
        if best_score < 0.65:
            best_answer = "I'm not sure about that. Ask me something else about space!"

        # ✅ Save the conversation
        chat = ChatMessage.objects.create(
            user_message=user_message,
            bot_reply=best_answer,
        )

        serializer = ChatMessageSerializer(chat)
        return Response(serializer.data, status=status.HTTP_200_OK)
