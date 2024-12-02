import xml.etree.ElementTree as ET
import os,shutil

def parse_xml(xml_file):
    tree = ET.parse(xml_file)    
    root = tree.getroot()
    data =[]
    for element in root.iter():
        data.append({element.tag:element.text})
    return data

def small_folder_process(folder_path,training_path,test_path): # 小型数据集分类
    if not os.path.exists(training_path):  
        os.makedirs(training_path)
    if not os.path.exists(test_path):  
        os.makedirs(test_path)
    files = os.listdir(folder_path) # 读入文件夹
    num = len(files) 
    training_num = 0.7*num # 区分训练集与测试集
    for file in files[:training_num+1]:
        output_path = os.path.join(training_path,os.path.basename(file))
        shutil.copy(file,output_path)
    for file in files[training_num+1:]:
        output_path = os.path.join(test_path,os.path.basename(file))
        shutil.copy(file,output_path)