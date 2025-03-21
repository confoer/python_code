from email import message
from turtle import pen
from vosk import Model, KaldiRecognizer
import json,os,cv2
import pyaudio
import pyttsx3
import ollama
import inference as inference
import os,dashscope


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

# def interact_with_ollama_model():
#     """使用 Ollama 本地大模型进行语音交互"""
#     speak("你好！有什么可以帮助你的吗？")
#     text = listen()
#     while True:
#         if keyboard.is_pressed('q'):
#             speak("再见！祝你有美好的一天！")
#             break
#         if "再见" in text:
#             speak("再见！祝你有美好的一天！")
#             break
#         if not text:
#             continue
#         else:
#             break
#         # 调用 Ollama 本地大模型的 API 进行对话
#     response = call_ollama_api(text)
#     speak(response)

def mood():
    if not os.path.exists("Match_Project\models\\best_model.pth"):
        raise FileNotFoundError("请先训练模型！")
    predictor = inference.EmotionPredictor("Match_Project\models\\best_model.pth")
    
    cap = cv2.VideoCapture(0)
    speak("按下空格键进行拍照")
    mood = inference.run_camera(predictor)
    return mood

def call_qianwen_api(text,mood):
    messages = [{'role': 'user', 'content':f'你是康复保健领域的专家，我现在出现这种状况：{text}，而且，我的心情是{mood},尽量通俗易懂与简洁，'}]
    response = dashscope.Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwq-32b", 
        messages=messages,
        stream=True,
    )
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    for chunk in response:
    # 如果思考过程与回复皆为空，则忽略
        if(chunk.output.choices[0].message.content == "" and chunk.output.choices[0].message.reasoning_content == ""):
            pass
        else:
            # 如果当前为思考过程
            if (chunk.output.choices[0].message.reasoning_content != "" and 
                chunk.output.choices[0].message.content == ""):
                print(chunk.output.choices[0].message.reasoning_content, end="",flush=True)
                reasoning_content += chunk.output.choices[0].message.reasoning_content
            # 如果当前为回复
            elif chunk.output.choices[0].message.content != "":
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(chunk.output.choices[0].message.content, end="",flush=True)
                answer_content += chunk.output.choices[0].message.content
    answer_content = answer_content.replace("**", "").replace("###","").replace("—", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    answer_content = answer_content.replace("\n", "")
    answer_content = " ".join(answer_content.split())  
    return answer_content

def call_ollama_api(text,mood):
    """调用 Ollama 本地大模型的 API，实现流式输出并统一格式"""
    print(f'提问：{text}')
    stream = ollama.generate(
        stream=False,
        model='deepseek-r1:8b',  # 指定模型名称
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
    speak("你好！有什么可以帮助你的吗？")
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
        # 调用 Ollama 本地大模型的 API 进行对话
    if not os.path.exists("Match_Project\models\\best_model.pth"):
        raise FileNotFoundError("请先训练模型！")
    predictor = inference.EmotionPredictor("Match_Project\models\\best_model.pth")
    cap = cv2.VideoCapture(0)
    speak("按下q进行拍照")
    moods = inference.run_camera(predictor)
    response = call_qianwen_api(text,moods)
    speak(response)