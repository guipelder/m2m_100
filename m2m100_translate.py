#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from flask import Flask, render_template, request, redirect, url_for
from langdetect import detect

app = Flask(__name__)

@app.route('/', methods=['GET'])
def detect_language():
    with open('input.txt', 'r') as file:
        #reading from the file in our case
        file_contents = file.read()
        #trying to detect the language but its not accurate
        #thats why there is manual translation option
        src_lang_detected = detect(str(file_contents))
    lang_code_src = src_lang_detected
    print(f"input.txt contents:\n{file_contents} \n language detected {src_lang_detected}")

    return f""" <p>  input.txt contents:\n{file_contents}$$$
                    \n --> language detected {src_lang_detected} </p>
                    <p>write the following address plus and language code
                    for manual translation if detected language is wrong.
                    Example:</p>
                    <a href="http://127.0.0.1:5000/{src_lang_detected}/fa">
                    http://127.0.0.1:5000/{src_lang_detected}/fa </a>
                    <p>or Just:</p>
                    <a href="http://127.0.0.1:5000/fa"> http://127.0.0.1:5000/fa </a>
 
                    """


@app.route('/<string:lang_code_src>/<string:lang_code_dest>', methods=['GET'])
def manual_language_code_translation(lang_code_src, lang_code_dest):
    with open('input.txt', 'r') as file:
        file_contents = file.read()
    print(f"input.txt contents:\n{file_contents} \n")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


    tokenizer.src_lang = lang_code_src
    encoded_text = tokenizer(file_contents, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, 
        forced_bos_token_id=tokenizer.get_lang_id(lang_code_dest))
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return output

#auto language translation
@app.route('/<string:lang_code_dest>', methods=['GET'])
def language_code_single_route( lang_code_dest):
    with open('input.txt', 'r') as file:
        file_contents = file.read()
        src_lang_detected = detect(str(file_contents))
    lang_code_src = src_lang_detected
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


    print(f"input.txt contents:\n{file_contents} \n language detected {src_lang_detected}")
    tokenizer.src_lang = lang_code_src
    encoded_text = tokenizer(file_contents, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, 
        forced_bos_token_id=tokenizer.get_lang_id(lang_code_dest))
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return  output

if __name__=="__main__":
    app.run(port=5000,debug=False)
