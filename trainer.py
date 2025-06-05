#
# Created By @TailsDev Or t.me/@Shoukaku07
# Initial Name: M
#
# Inspired GPT (OpenAI)
#

import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from tqdm import tqdm
from ai_modeling import TRForCausalLM

##########

model = TRForCausalLM()
model.load_state_dict(torch.load("./model/model.bin"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

##########

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

dataset = load_dataset("roneneldan/TinyStories", split="train")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

##########

batch_size = 8
num_samples = len(tokenized_dataset)
epochs = 3

# For calculate total steps!
#
# total_step = (num_samples * epochs) / batch_size
#
steps_per_epoch = num_samples // batch_size 
total_steps = steps_per_epoch * epochs


optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

##########

model.train()
loop = tqdm(range(total_steps), leave=True)

for step in loop:
    epoch = step // steps_per_epoch
    batch_idx = (step % steps_per_epoch) * batch_size

    optimizer.zero_grad()

    batch = tokenized_dataset[batch_idx : batch_idx + batch_size]
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']

    loss.backward()
    optimizer.step()
    scheduler.step()

    loop.set_description(f"Epoch {epoch+1}/{epochs}")
    loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

torch.save(model.state_dict(), "./outputs/model_pretrained.bin")
tokenizer.save_pretrained("./outputs")
