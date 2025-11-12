import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from groq import Groq
from PIL import Image
import streamlit as st
import io
import base64

# -----------------------------
# CONFIGURAÇÕES INICIAIS
# -----------------------------
st.set_page_config(layout="wide")
st.image("MathGestures.png")

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox("Rodar câmera", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Resposta")
    output_text_area = st.empty()

# -----------------------------
# CONFIGURAR CLIENTE GROQ VIA SECRETS
# -----------------------------
# No arquivo .streamlit/secrets.toml:
# [groq]
# api_key = "SUA_CHAVE_AQUI"

groq_api_key = st.secrets["groq"]["api_key"]
client = Groq(api_key=groq_api_key)

# -----------------------------
# CONFIGURAR CÂMERA
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(
    staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5
)

# -----------------------------
# FUNÇÕES
# -----------------------------
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    # desenha quando só o indicador está levantado
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)

    # limpa tela quando só o polegar está levantado
    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 0

    return current_pos, canvas


def sendToAI(canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "Você é um assistente de matemática. Resolva o problema mostrado nesta imagem "
            "e forneça apenas o resultado final, com uma pequena explicação."
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Você é um assistente de matemática útil."},
                {
                    "role": "user",
                    "content": f"{prompt}\n[Imagem codificada base64: data:image/png;base64,{image_base64}]",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    return None


# -----------------------------
# LOOP PRINCIPAL (STREAMLIT)
# -----------------------------
prev_pos = None
canvas = None
output_text = ""

if run:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        ai_text = sendToAI(canvas, fingers)
        if ai_text:
            output_text = ai_text

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.subheader(output_text)
else:
    st.warning("Ative a opção 'Rodar câmera' para iniciar.")
