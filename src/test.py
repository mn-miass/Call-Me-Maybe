from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

text = "What is the sum of 40 and 42?"
input_ids = model.encode(text)
print(input_ids)
