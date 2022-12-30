import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.train import Checkpoint
from nmt-french-eng import Encoder, Decoder



def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    sentence = preprocess_string(sentence)
    inputs = []
    for w in sentence.split(' '):
        if w in inp_lang.word_index.keys() :
            inputs.append(inp_lang.word_index[w])
    
    #inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    
def translate(sentence, eng_sen, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    print('Input: {}'.format(sentence))
    print('Actual translation: {}'.format(eng_sen))
    print('Predicted translation: {}'.format(result[:-7]))
    print("BLEU Score : {0:0.2f}".format(bleu_score.sentence_bleu(eng_sen, result[:-7], weights=(0.5, 0.5))))
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    plt.show()