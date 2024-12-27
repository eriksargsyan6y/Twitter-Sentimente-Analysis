import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#LOAD_DATA
train_data = pd.read_csv('LINK!')
validation_data = pd.read_csv('LINK!')
#DATA
X_tr = train_data['im getting on borderlands and i will murder you all ,']
y_tr = train_data['Positive']
y_tr = y_tr.map({'Positive': 1, 'Neutral': 0, 'Negative': -1, 'Irrelevant': -2})
X_val = validation_data['I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣']
y_val = validation_data['Irrelevant']
y_val = y_val.map({'Positive': 1, 'Neutral': 0, 'Negative': -1, 'Irrelevant': -2})
# DATA_CLEAR
X_tr = X_tr.fillna('').astype(str)
X_val = X_val.fillna('').astype(str)
#ONE_HOT_ENCODE
y_tr_categorical = to_categorical(y_tr, num_classes=4)
y_val_categorical = to_categorical(y_val, num_classes=4)
#TOKENIZER
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_tr)
X_train_sequences = tokenizer.texts_to_sequences(X_tr)
X_val_sequences = tokenizer.texts_to_sequences(X_val)
# PADDING
X_train_pad = pad_sequences(X_train_sequences, maxlen=100)
X_val_pad = pad_sequences(X_val_sequences, maxlen=100)
#MODELS
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(128, dropout=0.4, recurrent_dropout=0.4),
    Dense(4, activation='softmax')  
])
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# EarlyStopping 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
#MODEL FIT
history = model.fit(
    X_train_pad, y_tr_categorical,
    epochs=30,
    batch_size=64,
    validation_data=(X_val_pad, y_val_categorical)
)
#LOSS,ACCUARCY
test_loss, test_acc = model.evaluate(X_val_pad, y_val_categorical)
print(f'LOSS՝ {test_loss}')
print(f'ACCUARCY {test_acc}')
model.save('Models2.keras')
model.save('Models1.h5')
#TOKENIZATION
texts = train_data["im getting on borderlands and i will murder you all ,"].fillna("").astype(str)  
tokenizer = Tokenizer(num_words=10000)
#TOKENIZER_FIT
tokenizer.fit_on_texts(texts)
print("Dictionary:", tokenizer.word_index)
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#GRAFIC
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Los')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('train and val lossտ')
plt.show()
plt.plot(history.history['accuracy'], label='train accuarcyն')
plt.plot(history.history['val_accuracy'], label='val accuarcy')
plt.xlabel('epoch')
plt.ylabel('accuarcy')
plt.legend()
plt.title('train and val accuarcy')
plt.show()
