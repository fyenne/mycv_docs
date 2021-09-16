```yaml
http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture01-wordvecs1.pdf

http://web.stanford.edu/class/cs224n/

http://cs224d.stanford.edu/

http://cs224d.stanford.edu/lecture_notes/notes1.pdf

https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
https://www.kaggle.com/manithvazirani/transformers

https://www.tensorflow.org/tutorials/text/word2vec
```



nlp walkthrough


---



### eda

number of characters, number of word, avg word length

check common stop words. punctuations

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

def create_corpus(target):
    corpus=[]
    
    for x in tweet[tweet['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
...
## https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
```



#### easy to go sklean vectorizer, only one step to feed the model.

```python
train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])
```

#### or we go more detailed

word embeddings:

>transforming words into their real value vectors ==> vectorization
>
>for eg: word2Vec  & GloVe
>
>+ word2Vec shallow nn, to understand the probability of two or more words occurring together, thus to group words with similar meanings together to form a cluster in a vector space.
>
>  ![image-20210903154735903](C:\Users\dscshap3808\AppData\Roaming\Typora\typora-user-images\image-20210903154735903.png)
>
>+ > skip-gram model
>  >
>  > given a word, we’ll try to predict its neighboring words.
>
>+ > continuous bag of words (CBOW)
>  >surrounding words get together to predict middle.
>  > 
>
>  

![image-20210903152232603](C:\Users\dscshap3808\AppData\Roaming\Typora\typora-user-images\image-20210903152232603.png)

![image-20210903163727752](C:\Users\dscshap3808\AppData\Roaming\Typora\typora-user-images\image-20210903163727752.png)





















