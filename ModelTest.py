import pickle                                                   
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#LOAD_MODEL
model = load_model('LINK!')
#TOKENIZER_LOAD
with open('LINK!', 'rb') as file:
    tokenizer = pickle.load(file)
#Y_TARGET
class_mapping = {0: 'Neutral', 1: 'Positive', 2: 'Negative', 3: 'Irrelevant'}
#NEW_TEXT
new_texts = ["TEXT!!"]
#(tokenization  padding)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_pad_sequences = pad_sequences(new_sequences, maxlen=100)

predictions = model.predict(new_pad_sequences)

predicted_classes = predictions.argmax(axis=1)  

for text, prediction in zip(new_texts, predicted_classes):
    print(f"TEXT՝ {text} => CLASSIFICATION՝ {class_mapping[prediction]}")