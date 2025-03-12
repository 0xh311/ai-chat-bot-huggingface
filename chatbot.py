from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

while True:
    user_input = input("You: ")
    

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    

    bot_response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {bot_response}")
