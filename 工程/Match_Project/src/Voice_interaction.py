from email import message
from turtle import pen
from vosk import Model, KaldiRecognizer
import json,os,cv2
import pyaudio
import pyttsx3
import ollama
import inference as inference
import os,dashscope
import keyboard
import socket
import threading
import time
import requests

server_port="10005"
server_ip='116.62.59.66'
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
# Function to handle incoming connections on port 10005
def handle_client(client_socket, output_port):
    buffer = []
    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            buffer.append(data)
            print(f"Received: {data}")

            # Simulate waiting for a moment before sending to LLM
            time.sleep(2)

            # Process the buffered text with the LLM API
            text_to_process = ''.join(buffer)
            buffer.clear()
            llm_response = call_qianwen_api(text_to_process)

            # Send the LLM response to port 10006
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as output_socket:
                output_socket.connect(('127.0.0.1', output_port))
                output_socket.sendall(llm_response.encode('utf-8'))
                print(f"Sent to port {output_port}: {llm_response}")

        except Exception as e:
            print(f"Error: {e}")
            break

    client_socket.close()

def connect_to_server(message):
    
    try:
        # 发送消息给服务器
        client_socket.sendall(message.encode('utf-8'))
        # 接收服务器的响应
        response = client_socket.recv(1024).decode('utf-8')
        # 关闭套接字
        client_socket.close()
        return response

    except ConnectionRefusedError:
        print("连接被拒绝，请检查服务器是否正在运行，或者 IP 地址和端口是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")
    return None

# Main server function
def start_server(input_port=10005, output_port=10006):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', input_port))
    server.listen(5)
    print(f"Listening on port {input_port}...")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket, output_port))
        client_handler.start()

def speak(text):
    engine = pyttsx3.init()
    """将文字转换为语音输出"""
    engine.say(text)
    engine.runAndWait()

def listen():
    """监听麦克风输入并返回识别的文本"""
    model = Model("D:\\vosk-model-small-cn-0.22")
    recognizer = KaldiRecognizer(model, 16000)

    # 初始化音频流
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000
    )
    stream.start_stream()

    print("请说话...")
    final_result = ""
    try:
        while True:
            data = stream.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                print("最终识别结果：", result)
                result = json.loads(result)
                final_result = result.get("text", "")
                break
            else:
                partial_result = recognizer.PartialResult()
                print("部分识别结果：", partial_result)
    except KeyboardInterrupt:
        print("停止识别")

    # 清理资源
    stream.stop_stream()
    stream.close()
    p.terminate()

    final_result = final_result.replace(" ", "")

    return str(final_result)

def interact_with_ollama_model():
    """使用 Ollama 本地大模型进行语音交互"""
    speak("你好！有什么可以帮助你的吗？")
    text = listen()
    while True:
        if keyboard.is_pressed('q'):
            speak("再见！祝你有美好的一天！")
            break
        if "再见" in text:
            speak("再见！祝你有美好的一天！")
            break
        if not text:
            continue
        else:
            break
        # 调用 Ollama 本地大模型的 API 进行对话
    response = call_ollama_api(text)
    speak(response)

def mood():
    if not os.path.exists("./models\\est_model.pth"):
        raise FileNotFoundError("请先训练模型！")
    predictor = inference.EmotionPredictor("./models\\best_model.pth")
    speak("按下空格键进行拍照")
    mood = inference.run_camera(predictor)
    return mood

def call_qianwen_api(text,mood):
    response = f'1你是康复保健领域的专家，我现在出现这种状况：{text}，而且，我的心情是{mood},尽量通俗易懂与简洁，'
    return connect_to_server(response)

def call_ollama_api(text,mood):
    """调用 Ollama 本地大模型的 API，实现流式输出并统一格式"""
    print(f'提问：{text}')
    stream = ollama.generate(
        stream=False,
        model='deepseek-r1:8b',
        prompt=f'你是医疗保健领域的专家，我现在出现这种状况：{text}，而且，我的心情是{mood},请给我简单讲一讲，尽量通俗易懂与简洁，字数缩短到100字以内。'
    )
    print('-----------------------------------------')
    
    if isinstance(stream, ollama.GenerateResponse):
        full_response = stream.response
        if stream.done:
            full_response = full_response.replace("**", "").replace("###","").replace("—", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            full_response = full_response.replace("\n", "")
            full_response = " ".join(full_response.split())
            print(f'回答：{full_response}')
            return full_response
    else:
        full_response = ""
        for chunk in stream:
            if isinstance(chunk, dict):
                if 'response' in chunk:
                    full_response += chunk['response']
                elif 'message' in chunk and isinstance(chunk['message'], dict) and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
                if chunk.get('done'):
                    break
            elif isinstance(chunk, tuple):
                if len(chunk) > 0:
                    full_response += chunk[0]
                if len(chunk) > 1 and chunk[1]:
                    break
        print(f'回答：{full_response}')
        return full_response

    return ""

if __name__ == "__main__":
    print("你好！有什么可以帮助你的吗？")
    # text = listen()
    text = '头疼'
    while True:
        if "再见" in text:
            speak("再见！祝你有美好的一天！")
            break
        if not text:
            continue
        else:
            break
    if not os.path.exists("D:\Python_Code\Match_Project\models\\best_model.pth"):
        raise FileNotFoundError("请先训练模型！")
    predictor = inference.EmotionPredictor("D:\Python_Code\Match_Project\models\\best_model.pth")
    speak("按下空格键进行拍照")
    moods = inference.run_camera(predictor)
    response = call_qianwen_api(text,moods)
    
    print(response)