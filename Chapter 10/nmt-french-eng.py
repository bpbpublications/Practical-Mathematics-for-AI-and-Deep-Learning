import tensorflow as tf
from preprocess import get_data, lines, preprocess_string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.train import Checkpoint
import numpy as np
import matplotlib.pylab as plt

#Ref: https://medium.com/analytics-vidhya/seq2seq-models-french-to-english-translation-using-encoder-decoder-model-with-attention-9c05b2c09af8



class Encoder(tf.keras.Model):
    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform',
                                  recurrent_activation='sigmoid')
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    def initial_hidden_state(self):
        #Generating encoder initial states as all zeros
        return tf.zeros((self.batch_sz, self.enc_units))
    
    
    
    


class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.va = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        """
        query tensor of shape: [batch_size, hidden_size],
        value tensor of shape: [batch_size, inp_seq_len, hidden_size]
        """
        query = tf.expand_dims(query, 1)        
        scores = self.va(tf.nn.tanh(self.W1(query) + self.W2(values)))
        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return attention_weights, context_vector

    
#Decoder with attention
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size,
                 embedding_dim,
                 dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform',
                                  recurrent_activation='sigmoid')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)
       
    def call(self, x, hidden, enc_output):
        # enc_output (batch_size, max_length, hidden_size)
        # hidden (batch_size, hidden size)
        attention_weights, context_vector = self.attention(hidden,
                                                           enc_output)
        x = self.embedding(x)
        #Concatenating previous output with contx_vec
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return tf.zeros(self.batch_sz, self.dec_units)    

    
class NMTModel:
    def __init__(self,
                 vocab_size_in,
                 embedding_dim_in,
                 vocab_size_out,
                 embedding_dim_out,                 
                 enc_units,
                 dec_units,                                  
                 batch_sz,
                 inp_lang_tokenizer,
                 targ_lang_tokenizer
                ):
        super(NMTModel, self).__init__()
        self.inp_lang_tokenizer = inp_lang_tokenizer
        self.targ_lang_tokenizer = targ_lang_tokenizer
        
        self.encoder = Encoder(vocab_size_in + 1, embedding_dim_in, enc_units, batch_sz)
        self.decoder = Decoder(vocab_size_out + 1, embedding_dim_out, dec_units, batch_sz)    
        self.optimizer = tf.keras.optimizers.Adam() 
        self.checkpoint_dir = './FrenchToEnglish/CheckPoint'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = Checkpoint(optimizer=self.optimizer,
                                     encoder=self.encoder,
                                     decoder=self.decoder,
                                     step=tf.Variable(1))        
            
    def loss_function(self, real, pred):
        #do not evaluate on zero labels 
        mask = tf.equal(real, 0) 
        weights = 1. - tf.cast(mask, tf.float32)
        
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=real, logits=pred)*weights
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ):    
        loss = 0
        #Getting initial encoder states (all zeros)
        hidden = self.encoder.initial_hidden_state()
        with tf.GradientTape() as tape:
            enc_output , enc_hidden = self.encoder(inp, hidden)
            #Setting final encoder states as initial decoder states
            dec_hidden = enc_hidden
            #Teacher forcing: feeding the target as the next input
            #Passing '<start>' token as initial token
            dec_input = tf.expand_dims(
                [self.targ_lang_tokenizer.word_index['<start>']]*BATCH_SIZE, 1)
            for t in range(1, targ.shape[1]) :
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:,t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:,t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.variables + self.decoder.variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
    
    def train(self, dataset):
        EPOCH = 7    
        for epoch in range(EPOCH):
            total_loss = 0
            for batch, (inp, targ) in enumerate(dataset):
                loss = self.train_step(inp, targ)
                total_loss+= loss
                if batch % 2 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy()))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))  

    
    def test_step(self, inp,max_length_targ, max_length_inp):  
        result = ''
        attention_plot = np.zeros([max_length_targ, max_length_inp])  
        hidden = [tf.zeros((1,self.encoder.enc_units))]
        enc_out, enc_hidden = self.encoder(inp, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang_tokenizer.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            # storing the attention weigths to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))            
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.targ_lang_tokenizer.index_word[predicted_id] + ' '

            if self.targ_lang_tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, attention_plot

    def translate(self, sentence, max_length_targ, max_length_inp):
        #Load Model Weights
        status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()    
        print(status)
        sentence = preprocess_string(sentence)
        inputs = []
        for w in sentence.split(' ') :
            if w in self.inp_lang_tokenizer.word_index.keys() :
                inputs.append(self.inp_lang_tokenizer.word_index[w])

        #inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result, attention_plot = self.test_step(inputs, max_length_targ, max_length_inp)
        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))
        return result, attention_plot
    
    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        columns = len(sentence)+1
        rows = len(predicted_sentence)+1
        
        ax.matshow(attention[0:rows,0:columns], cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
        plt.show()
        plt.savefig('attention.png')
        
if __name__ == "__main__":
    #Get data
    french, english = get_data(lines)
    fre_tr, fre_te, eng_tr, eng_te = train_test_split(french, english, test_size = 0.2, random_state = 43)
    

    #Tokenizing and padding input language
    fre_token = Tokenizer(filters='', lower = False)
    fre_token.fit_on_texts(fre_tr)
    fre_tokenized = fre_token.texts_to_sequences(fre_tr)
    fre_padded = pad_sequences(fre_tokenized, padding='post')

    #Tokenizing and padding target language
    eng_token = Tokenizer(filters='', lower = False)
    eng_token.fit_on_texts(eng_tr)
    eng_tokenized = eng_token.texts_to_sequences(eng_tr)
    eng_padded = pad_sequences(eng_tokenized, padding='post')

    #Number of unique tokens in input and output languages
    num_ip_tokens = len(fre_token.word_index)   #French
    num_op_tokens = len(eng_token.word_index)   #English

    #Maximum length of a sentence in both the languages
    max_len_ip = fre_padded.shape[1]   #French
    max_len_op = eng_padded.shape[1]   #English
    
    
    import tensorflow as tf

    BUFFER_SIZE = len(fre_padded)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256
    units = 512

    dataset = tf.data.Dataset.from_tensor_slices((fre_padded, eng_padded)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
    nmt = NMTModel(
                 vocab_size_in = num_ip_tokens,
                 embedding_dim_in = embedding_dim,
                 vocab_size_out = num_op_tokens,
                 embedding_dim_out = embedding_dim,                 
                 enc_units = units,
                 dec_units = units,                                  
                 batch_sz = BATCH_SIZE,
                 inp_lang_tokenizer = fre_token,
                 targ_lang_tokenizer = eng_token)
    
    #nmt.train(dataset)
    result, attention_plot = nmt.translate('Je ne parle pas anglais.', max_len_op, max_len_ip)
    print(result)
    """
    encoder = Encoder(num_ip_tokens + 1, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(num_op_tokens + 1, embedding_dim, units, BATCH_SIZE)
    
    optimizer = tf.keras.optimizers.Adam()
 

    checkpoint_dir = './FrenchToEnglish/CheckPoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    EPOCH = 20    
    for epoch in range(EPOCH):
        total_loss = 0
        for batch, (inp, targ) in enumerate(dataset):
            loss = train_step(inp, targ)
            total_loss+= loss
            if batch % 2 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy()))
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))  
    """
    #Load Model Weights
    #status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()    
    #print(status)
    
    