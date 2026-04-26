from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import random

def generate_synthetic_dataset(num_examples=1000):
    """Generates a synthetic dataset of success and failure examples."""
    data = []
    topics = ["code", "research", "writing", "analysis"]
    
    for _ in range(num_examples):
        topic = random.choice(topics)
        is_failure = random.random() < 0.2
        
        if is_failure:
            mode = random.choice(["hallucination", "off_task", "incomplete"])
            if mode == "hallucination":
                completion = "The capital of France is Berlin and 2+2 is 5."
            elif mode == "off_task":
                completion = "I am a helpful assistant, but I would like to talk about cookies instead of your request."
            else:
                completion = "The process involves step 1 and then..." # Incomplete
        else:
            mode = None
            completion = f"This is a correct and high-quality response about {topic}."
            
        data.append({
            "prompt": f"Perform a {topic} task.",
            "completion": completion,
            "label": "failure" if is_failure else "success",
            "failure_mode": mode
        })
        
    return Dataset.from_list(data)

def train_sft():
    # Setup
    model_id = "HuggingFaceTB/SmolLM-135M"
    dataset = generate_synthetic_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    sft_config = SFTConfig(
        output_dir="./models/sft_worker_smollm",
        max_seq_length=512,
        dataset_text_field="completion",
        packing=False,
    )

    training_args = TrainingArguments(
        output_dir="./models/sft_worker_smollm",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting SFT fine-tuning...")
    trainer.train()
    
    trainer.save_model("./models/sft_worker_smollm_final")
    print("SFT complete. Model saved to ./models/sft_worker_smollm_final")

if __name__ == "__main__":
    train_sft()
