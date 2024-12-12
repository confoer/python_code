from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="semantic_segmentation")

output = pipeline.predict("makassaridn-road_demo.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出