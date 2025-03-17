from vosk import Model, KaldiRecognizer
import json
import pyaudio
import pyttsx3
import keyboard
import ollama

# 初始化
engine = pyttsx3.init()

def speak(text):
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
    interact_with_ollama_model()
    
def call_ollama_api(text):
    """调用 Ollama 本地大模型的 API，实现流式输出并统一格式"""
    print(f'提问：{text}')
    stream = ollama.generate(
        stream=False,
        model='deepseek-r1:8b',  # 指定模型名称
        prompt=text,
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
    interact_with_ollama_model()