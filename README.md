

# Model Card for Meta-Llama3-8B-Instruct-assessment 

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This is the model card of a Meta-Llama3-8B-Instruct-assessment model that has been developed by fine-tuning Meta-Llama3-8B-Instruct. The model is finetuned using LoRA. This model calculates the semantic similarity between student explanation and expert/standard explanation for a given line of code during source code comprehension.
- **Developed by:** xap
- **License:** llama3
- **Finetuned from model :** meta-llama/Meta-Llama-3-8B-Instruct
- **Finetuned using dataset :** [SelfCode2.0](https://zenodo.org/records/10912669)



## How to Get Started with the Model

The dataset or input for this model should be in the alpaca format. If the model is loaded without using `PeftModel,`it will only load the base pre-trained weights without the LoRA fine-tuning. To use the model for inference use the following code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("xap/Meta-Llama3-8B-Instruct-assessment")
base_model = AutoModelForCausalLM.from_pretrained("xap/Meta-Llama3-8B-Instruct-assessment")

lora_config = LoraConfig.from_pretrained("xap/Meta-Llama3-8B-Instruct-assessment")
model = PeftModel.from_pretrained(
    base_model,
    "xap/Meta-Llama3-8B-Instruct-assessment",
    lora_config=lora_config,
)

inputt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  ### Instruction: For the given line of code, both the student and expert have provided the explanation for that line of code. Compute the semantic similarity between the student explanation and the expert explanation for the line of code.. ###  Input: for given line of code int[] values = {5, 8, 4, 78, 95, 12, 1, 0, 6, 35, 46};, the expert explanation is We declare an array of values to hold the numbers. and the student explanation is This line creates the integer array with the values. you need this to achieve the goal bc you need an array to look in ### Response: "
inputs = tokenizer(inputt,return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```
@misc {jeevan_2024,
	author       = { {jeevan} },
	title        = { Meta-Llama3-8B-Instruct-assessment (Revision 726a029) },
	year         = 2024,
	url          = { https://huggingface.co/xap/Meta-Llama3-8B-Instruct-assessment },
	doi          = { 10.57967/hf/2244 },
	publisher    = { Hugging Face }
}
```
