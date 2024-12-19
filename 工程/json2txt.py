import json,os,sys
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QLabel, QPushButton


type_txt = {
    'class': "0",
    'people': "1"
}

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 400, 300)
        layout = QVBoxLayout()
        input_folder_path = QLabel('Input folder path:', self)
        output_folder_path = QLabel('Output folder path:', self)
        self.input_folder_path = QLineEdit(self)
        self.output_folder_path = QLineEdit(self)
        self.button = QPushButton('Enter', self)
        self.button.clicked.connect(self.on_button_clicked)
        layout.addWidget(input_folder_path)
        layout.addWidget(self.input_folder_path)
        layout.addWidget(output_folder_path)
        layout.addWidget(self.output_folder_path)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_button_clicked(self):
        self.process_files()

    def process_files(self):
        json_folder_path = self.input_folder_path.text().strip().strip('"')
        txt_folder_path = self.output_folder_path.text().strip().strip('"')
        if not json_folder_path or not txt_folder_path:
            print("Both input and output folder paths must be provided")
            return 
        if not os.path.isdir(json_folder_path):
            print(f"The input folder path'{json_folder_path}'is not a valid directory.")
            return
        self.create_directory(txt_folder_path)
        
        try:
            for filename in os.listdir(json_folder_path):
                if filename.endswith('.json'):
                    self.process_json_file(json_folder_path, txt_folder_path, filename)
            print(f"Data has been successfully extracted and saved in {txt_folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.close()

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def process_json_file(self, json_folder_path, txt_folder_path, filename):
        json_file_path = os.path.join(json_folder_path, filename)
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        if not data.get('outputs'):
            print(f"Skipping file {filename} because 'outputs' is empty.")
            return
        
        width = data['size']['width']
        height = data['size']['height']
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_file_path = os.path.join(txt_folder_path, txt_filename)
        
        with open(txt_file_path, 'w') as txt_file:
            for obj in data['outputs']['object']:
                class_name = obj['name']
                txt_class = type_txt.get(class_name, "unknown")  
                self.write_txt_file(txt_file, class_name, txt_class, width, height, obj)

    def write_txt_file(self, txt_file, class_name, txt_class, width, height, obj):
        bndbox = obj.get('bndbox')
        if bndbox is None:
            print(f"'bndbox' key is missing in object: {obj}")
            return 
        xmin = bndbox['xmin']
        ymin = bndbox['ymin']
        xmax = bndbox['xmax']
        ymax = bndbox['ymax']
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        txt_file.write(f"{txt_class} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()