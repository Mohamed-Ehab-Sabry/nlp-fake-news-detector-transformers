from transformers import BertTokenizer
import tensorflow as tf
import numpy as np

class BertDataPreparer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def _encode_batch(self, text, label):
        def tokenize_func(t):
            texts = [str(x.decode('utf-8')) for x in t.numpy()]
            
            encodings = self.tokenizer(texts, 
                                       truncation=True, 
                                       padding='max_length', 
                                       max_length=128, 
                                       return_tensors='np') 
            
            return encodings['input_ids'].astype(np.int32), \
                   encodings['attention_mask'].astype(np.int32), \
                   encodings['token_type_ids'].astype(np.int32)

        [ids, mask, types] = tf.py_function(tokenize_func, [text], [tf.int32, tf.int32, tf.int32])
        
        ids.set_shape([None, 128])
        mask.set_shape([None, 128])
        types.set_shape([None, 128])
        
        return {'input_word_ids': ids, 'input_mask': mask, 'input_type_ids': types}, label

    def create_dataset(self, texts, labels, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(texts), 10000))
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset.prefetch(tf.data.AUTOTUNE)