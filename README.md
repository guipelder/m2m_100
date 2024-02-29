# Language Translation API
This web application uses the `transformers` library from Hugging Face to perform language detection and translation of text files using the T5 model. The script also utilizes Flask framework for building the web interface.

## Getting Started
These instructions will help you set up this project on your local machine for development and testing purposes.

### Prerequisites
Ensure Python (version >= 3.6) is installed in your system. You may download it from https://www.python.org/downloads/. Also, make sure you have installed required libraries by running the command below:
### Model
Download your model from [hugginface](https://huggingface.co/google/madlad400-3b-mt)
```
pip install -r requirements.txt
```
however it was not necessary for the `madlad` model but 
we use language detection based on implementation available at https://github.com/Mimino666/langdetect. Installation guide is provided within their repository.

### Execution
Run the main application by executing the following command:
```
python m2m100_translate.py
```
Open a browser window and navigate to URL shown in console (usually `http://127.0.0.1:5000/`). If needed, append appropriate language codes to access specific features.

## Built With
* [Flask](https://flask.palletsprojects.com/) - A lightweight WSGI web application framework.
* [Transformers](https://huggingface.co/transformers/) - State-of-the-art Machine Learning for Natural Language Processing.

## Bugs

## Authors
* **Mehdi Hosseini**  - [ Github ](https://github.com/guipelder)


## License
This project is licensed under the MIT License - see the LICENSE.md file for details
