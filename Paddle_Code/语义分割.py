from paddlex import create_model
model = create_model("PP-LiteSeg-T")
output = model.predict("img\\makassaridn-road_demo.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output_test/")
    res.save_to_json("./output_test/res.json")