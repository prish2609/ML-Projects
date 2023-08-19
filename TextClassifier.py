#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
get_ipython().system('pip install tensorflow-hub')
get_ipython().system('pip install tensorflow-datasets')
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# In[2]:


train_data, validation_data, test_data = tfds.load(name="imdb_reviews",
                                                  split=('train[:60%]', 'train[:60%]', 'test'),
                                                  as_supervised = True)


# In[3]:


train_data


# In[4]:


train_example_batch, train_labels_batch = next(iter(train_data.batch(10)))


# In[5]:


train_example_batch


# In[6]:


train_labels_batch


# In[7]:


embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape = [], dtype = tf.string, trainable = True)


# In[8]:


hub_layer(train_example_batch[:3])


# In[9]:


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))


# In[10]:


model.summary()


# In[11]:


model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])


# In[12]:


history = model.fit(train_data.shuffle(10000).batch(100), epochs=25,
                   validation_data = validation_data.batch(100), verbose=1)


# In[14]:


results = model.evaluate(test_data.batch(100), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name,value))


# In[ ]:




