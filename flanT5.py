from transformers import T5Tokenizer, T5ForConditionalGeneration

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

checkpoint="google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)


input_ids = tokenizer("what is your name?", return_tensors="pt").inputk_ids
outputs = model.generate(input_ids, max_length=500, num_beams=10, early_stopping=True)

print(tokenizer.decode(outputs[0]))