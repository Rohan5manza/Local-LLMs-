#LaMini-LM 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch



# simple ext-2-text generation inference
checkpoint="./model/" # LaMini-Flan-T5-248

tokenizer= AutoTokenizer.from_pretrained(checkpoint)

base_model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto',torch_dtype=torch.float32)

pipe = pipeline('text2text-generation', 
                 model = base_model,
                 tokenizer = tokenizer,
                 max_length = 512, 
                 do_sample=True,
                 temperature=0.3,
                 top_p=0.95,
                 )



from rich import console
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from functools import reduce
from itertools import chain
from datetime import datetime

console=Console()
console.print("[bold yellow]Preparing for LaMini model...")

import textwrap
response=''
instruction=console.input("Ask LaMini: ")
start=datetime.now()
console.print("[red blink]Executing...")
console.print(f"[grey78]Generating answer to your question:[grey78] [green_yellow]{instruction}")
#instruction = 'Write a travel blog about a 3-day trip to The Philippines'
generated_text = pipe(instruction)
for text in generated_text:
  response += text['generated_text']

wrapped_text=textwrap.fill(response,100)
console.print(Panel(wrapped_text, title="LaMini Reply", title_align="center"))
stop = datetime.now()
elapsed = stop - start
console.rule(f"Report Generated in {elapsed}")
console.print(f"LaMini @ {datetime.now().ctime()}")  