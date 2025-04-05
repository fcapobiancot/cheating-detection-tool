# Online Exam Cheating Detection System 🕵️♂️

Sistema de detección de conductas sospechosas en exámenes en línea mediante análisis multimodal (video + audio) utilizando TensorFlow Lite.

## Características principales ✨
- Detección de 3 comportamientos visuales:
  - 👀 Mirada normal
  - 👁️ Mirada desviada
  - 🚫 Ausencia de persona
- Análisis de audio en tiempo real:
  - 🔊 Ruido de fondo
  - 🗣️ Conversación detectada
- Interfaz gráfica intuitiva
- Procesamiento en tiempo real optimizado

## Requisitos del sistema 💻
- Python 3.9
- Webcam y micrófono 
- Sistema operativo: Windows

## Instalación ⚙️

1. Clonar repositorio:
```bash
git clone https://github.com/tu-usuario/cheating-detection-tool.git
cd cheating-detection-tool

2.Instalar las dependencias (entorno virtual para que no hayan conflictos con otras versiones)
```py
pip install tensorflow==2.12.0 opencv-python==4.8.0.76 numpy==1.24.3 pillow==10.0.0 moviepy==1.0.3
```
3.Ejecutar la aplicación principal:
```bash
python src/cheatingDetectionTool.py
```