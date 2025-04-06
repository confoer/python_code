class Config:
    # 数据配置
    data_root = "D:\Datasets\data\processed"
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # 训练参数
    batch_size = 28
    lr = 0.001
    epochs = 20
    num_workers = 8
    step_size = 4
    gamma = 0.1
    
    # 模型参数
    model_name = "mobilenetv2"
    pretrained = True
    input_size = 224
    
    # 路径配置
    checkpoint_dir = "Match_Project\models\\"
    result_dir = "Match_Project\\results\\"

config = Config()