import os
import torch
import pickle
import xml.dom.minidom
from torch.utils.data import TensorDataset

def getData(type,tokenizer,file_name,max_length):
    if type=='train' or type=='eval':
        DOMTree = xml.dom.minidom.parse(file_name)
        collection = DOMTree.documentElement
        DOCS=collection.getElementsByTagName("Doc")
        data=[]
        for DOC in DOCS:
            doc=[]
            doc_content=''
            Sentences=DOC.getElementsByTagName('Sentence')
            for Sentence in Sentences:
                try:
                    if Sentence.hasAttribute('label'):
                        label=eval(Sentence.getAttribute('label'))
                        doc.append((Sentence.childNodes[0].data,label))
                except:
                    continue
                doc_content+=Sentence.childNodes[0].data
            [data.append((d,doc_content)) for d in doc]

        sents=[tokenizer(temp[0][0],padding='max_length',truncation=True,max_length=max_length) for temp in data]
        sents_input_ids=torch.tensor([temp["input_ids"] for temp in sents])
        sents_attn_masks=torch.tensor([temp["attention_mask"] for temp in sents])

        contexts=[tokenizer(temp[1],padding='max_length',truncation=True,max_length=max_length) for temp in data]
        contexts_input_ids=torch.tensor([temp["input_ids"] for temp in contexts])
        contexts_attn_masks=torch.tensor([temp["attention_mask"] for temp in contexts])

        labels=torch.tensor([temp[0][1] for temp in data])

        dataset=TensorDataset(sents_input_ids,sents_attn_masks,contexts_input_ids,contexts_attn_masks,labels)
        return dataset
    
    if type=='test':
        DOMTree = xml.dom.minidom.parse(file_name)
        collection = DOMTree.documentElement
        DOCS=collection.getElementsByTagName("Doc")
        data=[]
        for DOC in DOCS:
            doc=[]
            doc_content=''
            Sentences=DOC.getElementsByTagName('Sentence')
            D_ID=DOC.getAttribute('ID')
            for Sentence in Sentences:
                try:
                    if Sentence.hasAttribute('ID'):
                        S_ID=Sentence.getAttribute('ID')
                        temp_ID=D_ID+'-'+S_ID
                        doc.append((Sentence.childNodes[0].data,temp_ID))
                except:
                    continue
                doc_content+=Sentence.childNodes[0].data
            [data.append((d,doc_content)) for d in doc]
        sents=[tokenizer(temp[0][0],padding='max_length',truncation=True,max_length=256) for temp in data]
        sents_input_ids=torch.tensor([temp["input_ids"] for temp in sents])
        sents_attn_masks=torch.tensor([temp["attention_mask"] for temp in sents])

        contexts=[tokenizer(temp[1],padding='max_length',truncation=True,max_length=256) for temp in data]
        contexts_input_ids=torch.tensor([temp["input_ids"] for temp in contexts])
        contexts_attn_masks=torch.tensor([temp["attention_mask"] for temp in contexts])

        ids=[temp[0][1] for temp in data]

        dataset=TensorDataset(sents_input_ids,sents_attn_masks,contexts_input_ids,contexts_attn_masks)

    return dataset,ids

def getTrainData(tokenizer,bert_name,max_length=256):
    feature_file = "/Path_of_your_dataset/SMP/%s/%s/train_features.pkl"%((bert_name.split('/')[-1]),max_length)
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getData("train",tokenizer,'/Path_of_your_dataset/SMP/SMP2019_ECISA_Train.xml',max_length)
        os.mkdir( "/Path_of_your_dataset/SMP/%s/%s"%((bert_name.split('/')[-1]),max_length))
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)
    return train_dataset

def getEvalData(tokenizer,bert_name,max_length=256):
    feature_file = "/Path_of_your_dataset/SMP/%s/%s/eval_features.pkl"%((bert_name.split('/')[-1]),max_length)
    if os.path.exists(feature_file):
        eval_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        eval_dataset = getData("eval",tokenizer,'/Path_of_your_dataset/SMP/SMP2019_ECISA_Dev.xml',max_length)
        with open(feature_file, 'wb') as w:
            pickle.dump(eval_dataset, w)
    return eval_dataset

def getTestData(tokenizer,bert_name,max_length=256):
    feature_file = "/Path_of_your_dataset/SMP/%s/%s/test_features.pkl"%((bert_name.split('/')[-1]),max_length)
    feature_file_ids = "/Path_of_your_dataset/SMP/%s/%s/test_features_ids.pkl"%((bert_name.split('/')[-1]),max_length)
    if os.path.exists(feature_file):
        test_dataset = pickle.load(open(feature_file, 'rb'))
        ids=pickle.load(open(feature_file_ids,'rb'))
    else:
        test_dataset,ids = getData("test",tokenizer,'/Path_of_your_dataset/SMP/SMP2019_ECISA_Test.xml',max_length)
        with open(feature_file, 'wb') as w:
            pickle.dump(test_dataset, w)
        with open(feature_file_ids,'wb') as w:
            pickle.dump(ids,w)
    return test_dataset,ids