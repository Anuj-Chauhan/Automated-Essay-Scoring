#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import spacy
import string
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier

from textstat.textstat import textstatistics, easy_word_set, legacy_round 

import time
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_excel('/Users/anuj/Datasets/data_mining/asap-aes/training_set_rel3.xlsx')


# In[3]:


data.describe()


# In[ ]:


data.groupby('essay_set').agg('count')


# In[4]:


data_set1 = data[data['essay_set'] == 1]
data_set3 = data[data['essay_set'] == 3]
data_set5 = data[data['essay_set'] == 5]
data_set6 = data[data['essay_set'] == 6]


# In[5]:


data_set1.head(10).dropna(axis = 1)


# In[6]:


data_set1.essay[1]


# In[7]:


data_set1.dropna(axis = 1, inplace=True)
data_set3.dropna(axis = 1, inplace=True)
data_set5.dropna(axis = 1, inplace=True)
data_set6.dropna(axis = 1, inplace=True)


# In[8]:


#Validate if domain1_score is simply the sum of 2 scorers -- True
data_set1[data_set1['rater1_domain1'] + data_set1['rater2_domain1'] != data_set1['domain1_score']]

#Number of cases where the rater1 and rater2 scores don't match -- 618
len(data_set1[data_set1['rater1_domain1'] != data_set1['rater2_domain1']])

#Number of cases where rater1 and rater2 scores match -- 1165
len(data_set1) - len(data_set1[data_set1['rater1_domain1'] != data_set1['rater2_domain1']])


# In[9]:


nlp = spacy.load('en')


# In[10]:


#     kNN using word vectors (find the most similar documents and get a weighted score)
#     Simple linear regression using word counts, sentence length, number of distinct words and # verbs/nouns (as well as ratios/percentages of the pairs)
#     Boosted decision trees on the same features as above.
#     Multiclass SVM trained on the word vectors using the score as the "class"
#     Support vector regression trained on the word vectors using the score as target.
#     Singular value decomposition on the word vectors.
#     Linear combinations of all the above.

# Results
# Global parameters alone got me to around 0.71, 
# adding kNN got me to 0.74. 
# with the SVMs since I couldn't get past 0.75 with these features/algos.


# In[11]:


work_set = data_set1[['essay_id', 'essay', 'domain1_score']]
work_set_3 = data_set3[['essay_id', 'essay', 'domain1_score']]
work_set_5 = data_set5[['essay_id', 'essay', 'domain1_score']]
work_set_6 = data_set6[['essay_id', 'essay', 'domain1_score']]


# In[ ]:


data_set1[['essay', 'domain1_score']]


# In[ ]:


def addSentenceLength (row):
    return len(re.split(r' *[\.\?!][\'"\)\]]* *', row))


# In[ ]:


def returnSentences(text):
    return re.split(r' *[\.\?!][\'"\)\]]* *', text)


# In[ ]:


def addNounCount (row):
    count = 0
    for sentence in row.split('. '):
        doc = nlp(sentence)
        for token in doc:
            if (token.pos_ == 'NOUN'):
                count += 1

    return count


# In[ ]:


def addPropnCount (row):
    count = 0
    
    for sentence in row.split('. '):
        doc = nlp(sentence)
        for token in doc:
            if (token.pos_ == 'PROPN'):
                count += 1
    return count


# In[ ]:


def addVerbCount (row):
    count = 0
    
    for sentence in row.split('. '):
        doc = nlp(sentence)
        for token in doc:
            if (token.pos_ == 'VERB'):
                count += 1
    return count


# In[ ]:


def addAdjCount (row):
    count = 0
    
    for sentence in row.split('. '):
        doc = nlp(sentence)
        for token in doc:
            if (token.pos_ == 'ADJ'):
                count += 1
    return count


# In[ ]:


def addWordCount (row):
    return len(row.split())


# In[ ]:


def addDistinctWords (row):
    unstop = {word for word in row.split() if word not in nlp.Defaults.stop_words}
    return len(unstop)


# In[ ]:


def addReadabilityIndex (row):
    charCount = len(row.replace(" ", ""))
    wordCount = len(row.split())
    senCount = len(re.split(r' *[\.\?!][\'"\)\]]* *', row))
    
    return (4.71 * (charCount/wordCount) + 0.5 * (wordCount/senCount) - 21.43)


# In[ ]:


def syllableCount(row):
    row = row.lower()
    count = 0
    for word in row.split():
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
    return count


# In[ ]:


def syllables_count(word): 
    return textstatistics().syllable_count(word)


# In[ ]:


def avg_syllables_per_word(text): 
    syllable = syllables_count(text) 
    words = addWordCount(text) 
    ASPW = float(syllable) / float(words) 
    return legacy_round(ASPW,2)


# In[ ]:


def avg_sentence_length(text): 
    words = addWordCount(text) 
    sentences = addSentenceLength(text) 
    average_sentence_length = float(words / sentences) 
    return average_sentence_length 


# In[ ]:


def difficult_words(text): 
  
    # Find all words in the text 
    words = [] 
    sentences = returnSentences(text)
    for sentence in sentences: 
        words += [str(token) for token in sentence] 
  
    # difficult words are those with syllables >= 2 
    # easy_word_set is provide by Textstat as  
    # a list of common words 
    diff_words_set = set() 
      
    for word in words: 
        syllable_count = syllables_count(word) 
        if word not in easy_word_set and syllable_count >= 2: 
            diff_words_set.add(word) 
  
    return len(diff_words_set) 


# In[ ]:


def poly_syllable_count(text): 
    count = 0
    words = [] 
    sentences = returnSentences(text) 
    for sentence in sentences: 
        words += [token for token in sentence] 
      
  
    for word in words: 
        syllable_count = syllables_count(word) 
        if syllable_count >= 3: 
            count += 1
    return count 


# In[ ]:


def flesch_reading_ease(text): 
    """ 
        Implements Flesch Formula: 
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW) 
        Here, 
          ASL = average sentence length (number of words  
                divided by number of sentences) 
          ASW = average word length in syllables (number of syllables  
                divided by number of words) 
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - float(84.6 * avg_syllables_per_word(text)) 
        
    return legacy_round(FRE, 2) 


# In[ ]:


def gunning_fog(text): 
    per_diff_words = (difficult_words(text) / addWordCount(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words) 
    return grade 


# In[ ]:


def smog_index(text): 
    """ 
        Implements SMOG Formula / Grading 
        SMOG grading = 3 + ?polysyllable count. 
        Here,  
           polysyllable count = number of words of more 
          than two syllables in a sample of 30 sentences. 
    """
  
    if addSentenceLength(text) >= 3: 
        poly_syllab = poly_syllable_count(text) 
        
        SMOG = (1.043 * (30*(poly_syllab / addSentenceLength(text)))**0.5) + 3.1291
        
        return legacy_round(SMOG, 3) 
    else: 
        return 0


# In[ ]:


def dale_chall_readability_score(text): 
    """ 
        Implements Dale Challe Formula: 
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365 
        Here, 
            PDW = Percentage of difficult words. 
            ASL = Average sentence length 
    """
    words = addWordCount(text) 
    # Number of words not termed as difficult words 
    count = words - difficult_words(text) 
    if words > 0: 
  
        # Percentage of words not on difficult word list 
  
        per = float(count) / float(words) * 100
      
    # diff_words stores percentage of difficult words 
    diff_words = 100 - per 
  
    raw_score = (0.1579 * diff_words) + (0.0496 * avg_sentence_length(text)) 
      
    # If Percentage of Difficult Words is greater than 5 %, then; 
    # Adjusted Score = Raw Score + 3.6365, 
    # otherwise Adjusted Score = Raw Score 
  
    if diff_words > 5:        
  
        raw_score += 3.6365
          
    return legacy_round(raw_score, 3) 


# In[ ]:


# flesch_reading_ease
# gunning_fog
# smog_index
# dale_chall_readability_score


# In[ ]:


def featureExtractor(work_set):
    
    start = time.time()
    end = time.time()
    work_set['nounCount'] = work_set.essay.apply(addNounCount)
    #end = time.time()
    print("Noun count :", time.time() - end)
    
    work_set['propnCount'] = work_set.essay.apply(addPropnCount)
    print ("PropN count :", time.time() - end)
    end = time.time()
    
    work_set['verbCount'] = work_set.essay.apply(addVerbCount)
    print ("Verb count :", time.time() - end)
    end = time.time()
    
    work_set['adjCount'] = work_set.essay.apply(addAdjCount)
    print ("ADJ count :", time.time() - end)
    end = time.time()
    
    work_set['senCount'] = work_set.essay.apply(addSentenceLength)
    work_set['wordCount'] = work_set.essay.apply(addWordCount)
    work_set['distinctCount'] = work_set.essay.apply(addDistinctWords)
    work_set['syllableCount'] = work_set.essay.apply(syllableCount)
    work_set['avgSPerWord'] = work_set.essay.apply(avg_syllables_per_word)
    print ("Numerical count :", time.time() - end)
    end = time.time()
    
    work_set['readabilityIndex'] = work_set.essay.apply(addReadabilityIndex)
    print ("SRI count :", time.time() - end)
    end = time.time()

    work_set['riFRE'] = work_set.essay.apply(flesch_reading_ease)
    print ("FRE count :", time.time() - end)
    end = time.time()
    
    work_set['riGF'] = work_set.essay.apply(gunning_fog)
    print ("GF count :", time.time() - end)
    end = time.time()
    
    work_set['riSI'] = work_set.essay.apply(smog_index)
    print ("SI count :", time.time() - end)
    end = time.time()
    
    work_set['riDC'] = work_set.essay.apply(dale_chall_readability_score)
    print ("DC count :", time.time() - end)
    end = time.time()
    
    print ("Finished feature extraction in:", time.time() - start)
    print ()
    print ()


# In[ ]:


featureExtractor(work_set)


# In[ ]:


featureExtractor(work_set_3)


# In[ ]:


featureExtractor(work_set_5)


# In[ ]:


featureExtractor(work_set_6)


# In[ ]:


# work_set.to_pickle("./work_set_feats.pkl")
# work_set_3.to_pickle("./work_set_3_feats.pkl")
# work_set_5.to_pickle("./work_set_5_feats.pkl")
# work_set_6.to_pickle("./work_set_6_feats.pkl")


# In[34]:


work_set = pd.read_pickle("./work_set_feats.pkl")
work_set_3 = pd.read_pickle("./work_set_3_feats.pkl")
work_set_5 = pd.read_pickle("./work_set_5_feats.pkl")
work_set_6 = pd.read_pickle("./work_set_6_feats.pkl")


# In[ ]:


work_set.drop(columns= ['essay_id', 'essay'], axis = 1)


# In[ ]:


X = work_set_6.drop(['domain1_score','essay','essay_id'], 1)
y = work_set_6['domain1_score']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=7)


# In[ ]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print("R2 score on the Train:\t{:0.3f}".format(r2_score(y_train, classifier.predict(X_train))))
print("R2 score on the Test:\t{:0.3f}".format(r2_score(y_test, classifier.predict(X_test))))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

print("R2 score on the Train:\t{:0.3f}".format(r2_score(y_train, knn.predict(X_train))))
print("R2 score on the Test:\t{:0.3f}".format(r2_score(y_test, knn.predict(X_test))))


# In[ ]:


X = work_set_3.drop(['domain1_score','essay','essay_id'], 1)
y = work_set_3['domain1_score']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=7)


# In[ ]:


scaled_X_train = preprocessing.scale(X_train)
scaled_y_train = preprocessing.scale(y_train)
scaled_X_test = preprocessing.scale(X_test)
scaled_y_test = preprocessing.scale(y_test)


# In[ ]:


def build_model():
    model = keras.Sequential([
        layers.Dense(72, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(48, activation=tf.nn.relu),
        layers.Dense(1)
      ])
    
    goldenValue = 0.001
    optimizer = tf.keras.optimizers.RMSprop(goldenValue)
    
    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('Processed {} epochs.'.format(epoch))

EPOCHS = 40

history = model.fit(scaled_X_train, scaled_y_train, epochs=EPOCHS+1, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# In[ ]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:


# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(scaled_y_train, model.predict(scaled_X_train))))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(scaled_y_test, model.predict(scaled_X_test))))


# In[ ]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Domain Score]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Domain Score^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(history)


# In[ ]:


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(scaled_X_train, scaled_y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# In[ ]:


test_predictions = model.predict(scaled_X_train).flatten()

plt.scatter(scaled_y_train, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[ ]:


work_set = pd.read_pickle('./work_set.pkl')


# In[36]:


def generateMetaNeural(df, fname = "metafile"):
    df.pop('essay')
    columns = [key for key in df.keys() if (key  != 'essay_id') and (key != 'domain1_score')]
    columns.append('domain1_score')
    columns.append('essay_id')
    
    df[columns].to_csv(fname+'.csv', header=False, index=False)
    return 


# In[14]:


work_set.head()


# In[20]:


work_set.pop('essay')


# In[21]:


columns = [key for key in work_set.keys() if (key  != 'essay_id') and (key != 'domain1_score')]


# In[22]:


columns.append('domain1_score')
columns.append('essay_id')


# In[24]:


work_set[columns].head()


# In[37]:


generateMetaNeural(work_set, 'set1')


# In[38]:


generateMetaNeural(work_set_3, 'set3')


# In[39]:


generateMetaNeural(work_set_5, 'set5')


# In[40]:


generateMetaNeural(work_set_6, 'set6')


# In[ ]:




