from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="table_recognition")
output = pipeline.predict("table_recognition.jpg")
for res in output:
    # res.print() ## 打印预测的结构化输出
    # res.save_to_img("D:/Python/img") ## 保存img格式结果
    res.save_to_xlsx("D:/Python/img") ## 保存表格格式结果
    # res.save_to_html("D:/Python/img") ## 保存html结果
