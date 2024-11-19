from pprint import pprint
from paddlenlp import Taskflow

schema = ['选手成绩']
ie = Taskflow('information_extraction', schema=schema, model='uie-base')
results = ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
extracted_entities = []
for entity_type, mentions in results[0].items():
    for mention in mentions:
        extracted_entities.append({entity_type,mention['text']})
pprint(extracted_entities)

## 关系抽取
# schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
# ie = Taskflow('information_extraction', schema=schema, model='uie-base')
# ie.set_schema(schema) # Reset schema
# results = ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。')
# extracted_texts = []
# for result in results:
#     entity_type = list(result.keys())[0]  
#     entity_info = result[entity_type][0] 
#     extracted_texts.append({entity_type,entity_info['text']})
#     # 提取关系的文本
#     for relation_type, relations in entity_info['relations'].items():
#         for relation in relations:
#             extracted_texts.append({relation_type: relation['text']})
# pprint(extracted_texts)

## 事件抽取
# schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']} 
# ie = Taskflow('information_extraction', schema=schema, model='uie-base')
# ie.set_schema(schema)
# results = ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
# extracted_texts = []
# for result in results:
#     entity_type = list(result.keys())[0]  
#     entity_info = result[entity_type][0] 
#     for relation_type, relations in entity_info['relations'].items():
#         for relation in relations:
#             extracted_texts.append({relation_type: relation['text']})
# pprint(extracted_texts)

## 观点抽取
# schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} 
# ie = Taskflow('information_extraction', schema=schema, model='uie-base')
# ie.set_schema(schema) 
# results = ie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队")
# extracted_info = []
# for entity_info in results:
#     for entity_type, entity_details in entity_info.items():
#         for detail in entity_details:
#             info_dict = {'实体文本': detail['text']}
#             if '情感倾向[正向，负向]' in detail['relations']:
#                 for relation in detail['relations']['情感倾向[正向，负向]']:
#                     info_dict['情感倾向[正向，负向]'] = relation['text']
#             if '观点词' in detail['relations']:
#                 for relation in detail['relations']['观点词']:
#                     info_dict['观点词'] = relation['text']
#             extracted_info.append(info_dict)
# pprint(extracted_info)

## 情感分类
# schema = '情感倾向[正向，负向]' 
# ie = Taskflow('information_extraction', schema=schema, model='uie-base')
# ie.set_schema(schema) 
# results = ie('这个产品用起来真的很流畅，我非常喜欢')
# extracted_info = []
# for entity_info in results:
#     for entity_type, entity_details in entity_info.items():
#         for detail in entity_details:
#             info_dict = {detail['text']}
#             extracted_info.append(info_dict)
# pprint(extracted_info)