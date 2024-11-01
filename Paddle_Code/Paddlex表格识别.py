from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存img格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
    res.save_to_html("./output/") ## 保存html结果