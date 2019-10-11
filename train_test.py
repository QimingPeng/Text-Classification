import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from utils import WarmupLinearSchedule, macro_f1

def train_eval(train_dataloader, dev_dataloader, model, ckpt_path, 
                train_steps, check_step, eval_step, lr, warmup_steps, cv_i):
    """训练模型"""

    ckpt_path = os.path.join(ckpt_path, "pytorch_model_{}.pkl".format(cv_i))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(model.named_parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps, train_steps)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    best_step = 0
    best_f1 = 0
    train_loss = 0

    bar = range(train_steps)
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        batch_tensor, batch_sent_len, batch_labels = batch

        output = model([batch_tensor, batch_sent_len])
        loss = loss_func(output, batch_labels)
        optimizer.zero_grad()

        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()
        scheduler.step()

        if (step+1) % check_step == 0:
            logging.info("Step: {}, Train_batch_loss: {}".format(step+1, train_loss/check_step))
            train_loss = 0

        if (step + 1) % eval_step == 0:
            model.eval()
            output_labels = []
            gold_labels = []
            with torch.no_grad():
                for dev_step, (dev_tensor, dev_sent_len, dev_labels) in enumerate(dev_dataloader):
                    dev_tensor = dev_tensor.to(device)
                    dev_sent_len = dev_sent_len.to(device)
                    dev_labels = dev_labels.to(device)

                    output = model([dev_tensor, dev_sent_len])
                    
                    output = output.to('cpu').numpy()
                    output_labels.append(output)
                    gold = dev_labels.to('cpu').numpy()
                    gold_labels.append(gold)

            output_labels = np.concatenate(output_labels, 0)
            gold_labels = np.concatenate(gold_labels, 0)
            dev_f1_score = round(macro_f1(output_labels, gold_labels), 4)
            
            if dev_f1_score > best_f1:
                best_step = step
                best_f1 = dev_f1_score
                torch.save(model.state_dict(), ckpt_path)

            logging.info("Dev_f1_score: {}, Best_dev_f1: {}\n".format(dev_f1_score, best_f1))

        if step + 1 - best_step > 3000:
            logging.info("Early stopped at Step: {}, Best_dev_f1: {}\n".format(step+1, best_f1))
            break
        model.train()
    return best_f1


def test(test_dataloader, model, device, dev_batch_size):
    model.eval()
    outputs = []
    gold_labels = []
    with torch.no_grad():
        for _, (batch_tensor, batch_sent_len, batch_labels) in enumerate(test_dataloader):
            batch_tensor = batch_tensor.to(device)
            batch_sent_len = batch_sent_len.to(device)
            batch_labels = batch_labels.to(device)

            output = model([batch_tensor, batch_sent_len])

            output = output.to('cpu').numpy()
            outputs.append(output)
            gold = batch_labels.to('cpu').numpy()
            gold_labels.append(gold)

    outputs = np.concatenate(outputs, 0)
    gold_labels = np.concatenate(gold_labels, 0)
    test_f1_score = round(macro_f1(outputs, gold_labels), 4)

    return outputs, test_f1_score