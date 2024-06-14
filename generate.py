from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import json
import pickle
import pdb
import argparse

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def main():
    parser = argparse.ArgumentParser(description="GPT gen script")

    # Add arguments
    parser.add_argument(
        "-b", "--benchmark", type=str, required=False, help="Name of benchmark dataset"
    )
    parser.add_argument(
        "-t", "--train_author_key", type=str, required=False, help="Author key in pkl file"
    )

    # Execute the parse_args() method
    args = parser.parse_args()

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    base_model = PeftModel.from_pretrained(
        base_model, f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto/ditto"
    )
    
    base_model.eval()
    
    generator = pipeline(
        "text-generation",
        model=base_model,
        device="cuda",
        tokenizer=tokenizer
    )
    
    generator.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    path = f"./benchmarks/{args.benchmark}/processed/{args.benchmark}_test.pkl"
    
    with open(path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    
    spec_dataset = data[int(args.train_author_key)]    

    tasks = []
    
    for item in spec_dataset:
        tasks.append([
            {
                "content": item["prompt"],
                "role": "user"
            }
        ])

    for task in tasks:
        
        outs = generator(
            task, 
            max_new_tokens=1024, do_sample=True, 
            temperature=1,
            num_return_sequences=10,
            return_full_text=False
        )


        for out in outs:
            print("SAMPLE: ")
            print(out["generated_text"])
            print()

if __name__ == "__main__":
    main()
    
