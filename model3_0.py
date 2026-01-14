import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

MODEL_DIR = r"C:\Users\KIIT\OneDrive\Desktop\projects\chatbot_model3"

assert os.path.exists(MODEL_DIR), "Model directory not found"
print("Model directory contents:")
print(os.listdir(MODEL_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("EOS token:", tokenizer.eos_token)
print("EOS token id:", tokenizer.eos_token_id)

def generate_reply(user_text, max_new_tokens=80):
    prompt = f"User: {user_text} Bot:"

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Bot:" in decoded:
        reply = decoded.split("Bot:")[-1]
    else:
        reply = decoded

    return reply.strip()

print("\nChatbot ready.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("Exiting chat.")
        break

    bot_response = generate_reply(user_input)
    print("Bot:", bot_response)
