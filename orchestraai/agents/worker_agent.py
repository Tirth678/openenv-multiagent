import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import time
from typing import Dict, Optional

class WorkerAgent:
    """Wrapper for Worker LLMs with simulation and real LLM modes."""
    
    def __init__(self, worker_config: Dict, use_real_llm: bool = False):
        self.config = worker_config
        self.use_real_llm = use_real_llm
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        if self.use_real_llm:
            self._init_real_llm()

    def _init_real_llm(self):
        model_id = self.config['model_id']
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> Dict:
        """Generates output, either simulated or from a real LLM."""
        start_time = time.time()
        
        if self.use_real_llm:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_used = len(outputs[0])
        else:
            # Simulation mode
            time.sleep(0.1) # Simulate latency
            output_text = f"This is a simulated response for the task: {prompt[:50]}..."
            tokens_used = len(output_text.split()) * 1.3 # Rough estimate
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "output": output_text,
            "tokens_used": int(tokens_used),
            "latency_ms": latency
        }

    def simulate_failure(self, output: str, failure_mode: str) -> str:
        """Corrupts output text to simulate specific failure modes."""
        if not failure_mode:
            return output
            
        words = output.split()
        if failure_mode == "hallucination":
            # Replace 30% of words with plausible wrong words
            for i in range(len(words)):
                if random.random() < 0.3:
                    words[i] = random.choice(["WRONG", "FALSE", "INCORRECT", "BOGUS", "FICTIONAL"])
            return " ".join(words)
            
        elif failure_mode == "off_task":
            # Prepend unrelated topic
            prefix = "Let me tell you about the history of the toaster before I answer your question. The first toaster was invented in... "
            return prefix + output
            
        elif failure_mode == "incomplete":
            # Truncate at 40-60%
            cutoff = int(len(words) * random.uniform(0.4, 0.6))
            return " ".join(words[:cutoff]) + " [TRUNCATED]"
            
        return output

    def get_state(self) -> Dict:
        """Returns the current state of the worker for UI."""
        return {
            "id": self.config['id'],
            "name": self.config['name'],
            "skill_level": self.config['skill_level'],
            "status": "READY"
        }
