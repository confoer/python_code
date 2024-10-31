from flask import Flask, render_template, request,redirect,url_for,flash,send_from_directory,jsonify,flash
from wordcloud import WordCloud
from docx import Document
from pdf2docx import Converter
from docx2pdf import convert
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR,draw_ocr
import pandas as pd
import traceback,pdfkit
import erniebot
import pymysql
import win32api,win32con
import os,sys

## 初始化
app = Flask(__name__)
erniebot.api_type = 'aistudio'  
erniebot.access_token = 'c45a630b4d316eae8d412079a5c73685927aedba' 
app.secret_key = 'secret'  
app.config['SECRET_KEY'] = 'secret'  
UPLOAD_FOLDER = './uploads/'  
DOWNLOAD_FOLDER = './downloads/'  
app.config['ALLOWED_EXTENSIONS'] = ['png','jpg', 'jpeg']

##检查是否存在文件夹
if not os.path.exists(UPLOAD_FOLDER):  
    os.makedirs(UPLOAD_FOLDER)  
if not os.path.exists(DOWNLOAD_FOLDER):  
    os.makedirs(DOWNLOAD_FOLDER) 

## 设置模型参数
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   
folder_dir = os.path.dirname(os.path.realpath(__file__))
paddleOCR_dir = os.path.join(folder_dir, "PaddleOCR")
cls_model_dir="./PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer/" 
det_model_dir="./PaddleOCR/inference/ch_ppocr_mobile_v2.0_det_infer/"
rec_model_dir="./PaddleOCR/inference/ch_ppocr_mobile_v2.0_rec_infer/" 
use_angle_cls = True
use_space_char = True
use_gpu = False 
cls = True

