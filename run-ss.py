import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy
from transformers import AutoModel, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
# model_checkpoint = "fnlp/bart-large-chinese"
# model_checkpoint = "./output1"
# model_checkpoint = "./outputner"
model_checkpoint = "./outputprompt"
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,BartForConditionalGeneration

# model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
# from model_bart3 import BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
print('start')
from model_bart3 import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained("./outputprompt")
# model = AutoModel.from_pretrained(model_checkpoint)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# text = "去年底胸痛,咯痰带血,在当地医院查胸片示：右下段肺转移可能,右肺中段病灶伴胸腰增厚粘连,目前苦胸痛,咳嗽,咯痰有时带血,夜晚口干苦,动则气喘，"
# text = "代诉：住院化疗稍恶心呕吐,厌食,右侧腹胀,上腹有肿胀感,化疗后大便排出鲜血,量不多,下肢肿，"
# text = "代诉：化疗，恶心，呕吐,纳差,右侧腹胀,上腹胀,大便带鲜血,血量不多,下肢肿"
print('start')
# text = '症状：目前,自觉症状不显,肝区不痛,无明显疲劳,大便时溏,月事正常,面色欠华,呈贫血貌。'
# model_input = tokenizer([text])
# input_ids = torch.LongTensor(model_input['input_ids'])
# translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,num_beams=1,num_return_sequences=1,max_length=500)
# result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
# herb_predict = result[0].replace(" ",'').replace(':',",").replace('，',',').replace('太太子参','太子参').replace(')',',').replace('(',',').replace('南沙参参','南沙参').replace('赤赤芍','赤芍').split(',')
# print(herb_predict)
# with open('zzzzz.txt','r',encoding='utf-8') as f:
#     data = f.readlines()
# with open('zzzzzcpt.txt','w') as f:
#     for i in data:
#         text = i.replace("\n","")
#         model_input = tokenizer([text])
#         input_ids = torch.LongTensor(model_input['input_ids'])
#         translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,num_beams=1,num_return_sequences=1,max_length=500)
#         result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
#         print(result[0].replace(" ",''))
#         f.write(text+"&*"+result[0].replace(" ",'')+'\n')
print('ok')
with open('zh==.txt','r',encoding='utf-8') as f:
    data = f.readlines()
pre = 0
rec = 0
f = 0
count_5 = 0
count_10 = 0
count_15 = 0
count = 0
all_count = 0

data = ["0,求证候：症状：胰体占位,肝转移,最近在上腹肝胆区疼痛明显,大便3－4日不行,恶心,纳差,口干苦,腹胀,,黄腻质暗紫,细&*oo"]
for i in data:
    i = i.replace('0,"',"")
    text = i.split('&*')[0]
    herb_true = i.split('&*')[1].replace(',"\n',"").replace('\n',"").replace('"',"").split(',')
    print(text)
    print(herb_true)
    model_input = tokenizer([text])
    input_ids = torch.LongTensor(model_input['input_ids'])
    translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,num_beams=3,num_return_sequences=1,max_length=500)
    result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
    herb_predict = result[0].replace(" ",'').replace(':',",").replace('，',',').replace('太太子参','太子参').replace(')',',').replace('(',',').replace('南沙参参','南沙参').replace('赤赤芍','赤芍').split(',')
    print(result)
#     try:
#         herb_predict.remove('')
#     except:
#         pass
#     print(herb_predict)
#     herb = []
#     for a in herb_true:
#         for b in herb_predict:
#             if a == b :
#                 herb.append(a)
#     print(herb)
#     print("pre",len(herb)/len(herb_true),"rec",len(herb)/len(herb_predict))
#     count += 1
#     pree = len(herb)/len(herb_true)
#     recc = len(herb)/len(herb_predict)
#     pre += pree
#     rec += recc
#     try:
#         f += 2*pree*recc/(pree+recc)
#     except:
#         pass
#     print(count)
#     all_count += len(herb)
#     if len(herb) >=5 :
#         count_5 += 1
#     if len(herb) >=10 :
#         count_10 += 1
#     if len(herb) >=15 :
#         count_15 += 1
# print('pre:',pre/count)
# print('rec:',rec/count)
# print('f:',f/count)
# print('5',count_5/count)
# print('10',count_10/count)
# print('15',count_15/count)
# print('平均',all_count/count)