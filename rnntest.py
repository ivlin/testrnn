#import numpy as np
#import tensorflow as tf

with open("lolita_part1.txt","r",encoding="utf-8") as file:
    text=file.read()
    
print(str(len(text)))