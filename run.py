import os
import sys
import time
import torch
import logging
import argparse
import datetime
import numpy as np
import torch.nn as nn
from util.fgm import FGM
from util.model import CONTEXT_BERT
# from util.plotPng import plotPng
from util.labelSmooth import LabelSmoothingCrossEntropy
from sklearn.metrics import classification_report
from util.dataProcess import getTrainData,getEvalData,getTestData
from transformers import BertModel,BertTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids",
                        default='0,1,2,3,4,5,6,7',
                        type=str)
    parser.add_argument("--max_length",
                        default=256,
                        type=int)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int)
    parser.add_argument("--test_batch_size",
                        default=128,
                        type=int)
    parser.add_argument("--warmup_prop",
                        default=0.1,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=int)
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--fgm',
                        type=bool,
                        default=True)                    
    parser.add_argument('--test',
                        type=bool,
                        default=False)
    parser.add_argument('--labelSmooth',
                        type=bool,
                        default=True)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    bert_name = 'hfl/chinese-bert-wwm-ext' #['hfl/chinese-bert-wwm-ext','hfl/chinese-roberta-wwm-ext-large']
    bert = BertModel.from_pretrained(bert_name, return_dict=False)
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    train_data=getTrainData(tokenizer,bert_name,args.max_length)
    eval_data=getEvalData(tokenizer,bert_name,args.max_length)
    test_data,ids=getTestData(tokenizer,bert_name,args.max_length)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d distributed training",device, n_gpu)

    model=CONTEXT_BERT(bert)
    torch.cuda.empty_cache()
    model.to(device)
    model = torch.nn.DataParallel(model)
    if args.labelSmooth:
        criterion = LabelSmoothingCrossEntropy(0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = args.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    total_steps = len(train_dataloader)*args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
                    
    
    loss_plot,acc_plot=[],[]

    for epoch_i in range(args.num_train_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.num_train_epochs))
        print('Training...')
        
        t0 = time.time()
        model.train()
        n_correct, n_total, loss_total = 0, 0, 0
        for step,batch in enumerate(train_dataloader):

            a_input_ids  = batch[0].to(device)
            a_input_mask = batch[1].to(device)
            b_input_ids  = batch[2].to(device)
            b_input_mask = batch[3].to(device)
            labels       = batch[4].to(device)
            
            optimizer.zero_grad()
                
            inputs=a_input_ids,a_input_mask,b_input_ids,b_input_mask 
            predict=model(inputs)
            loss=criterion(predict,labels)
            loss.backward()

            if args.fgm:
                fgm = FGM(model)
                fgm.attack()
                predict_adv=model(inputs)
                loss_adv=criterion(predict_adv,labels)
                loss_adv.backward()
                fgm.restore()
    
            optimizer.step()
            scheduler.step()
            
            n_correct += (torch.argmax(predict, -1) == labels).sum().item()
            n_total += len(predict)
            loss_total += loss.item() * len(predict)

            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))        
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                loss_plot.append(train_loss)
                acc_plot.append(train_acc)
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))   
                n_correct, n_total, loss_total = 0, 0, 0
                
        true_labels,predict_labels=[],[]  
        model.eval()      
        for batch in eval_dataloader:
            a_input_ids  = batch[0].to(device)
            a_input_mask = batch[1].to(device)
            b_input_ids  = batch[2].to(device)
            b_input_mask = batch[3].to(device)
            labels       = batch[4]
            inputs=a_input_ids,a_input_mask,b_input_ids,b_input_mask
            with torch.no_grad():        
                outputs = model(inputs)        
            predict=outputs.detach().cpu().numpy()
            predict=np.argmax(predict, axis=1).flatten()
            predict_labels.append(predict)
            true_labels.append(labels)
        true_labels=[y for x in true_labels for y in x]
        predict_labels=[y for x in predict_labels for y in x]
        print(classification_report(true_labels,predict_labels,digits=4))

    # plotPng(loss_plot,acc_plot)

    if args.test:
        model.eval()
        predict_labels=[]
        for batch in test_dataloader:
            a_input_ids  = batch[0].to(device)
            a_input_mask = batch[1].to(device)
            b_input_ids  = batch[2].to(device)
            b_input_mask = batch[3].to(device)
            inputs=a_input_ids,a_input_mask,b_input_ids,b_input_mask

            with torch.no_grad():        
                outputs = model(inputs)        
            predict=outputs.detach().cpu().numpy()
            predict=np.argmax(predict, axis=1).flatten()
            predict_labels.append(predict)
        predict_labels=[y for x in predict_labels for y in x]
        with open('Path_to_store_your_result','wb') as f:
            for (temp_id,label) in zip(ids,predict_labels):
                f.write(temp_id+'\t'+str(label)+'\n')
        

if __name__ == '__main__':
    main()