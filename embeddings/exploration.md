# Word Embeddings Exploration

## Learning Notes

TF-IDF represents words as separate features. This means words like “good” and “excellent” are treated as different words with no connection, even though they have similar meanings. TF-IDF can work well, but it does not understand meaning or similarity between words.

Word embeddings solve part of this problem by representing each word as a dense vector. Similar words are placed close to each other in the vector space. For example, positive words like “good”, “great”, and “excellent” should be close together.

Word2Vec learns word vectors from context. CBOW predicts a word from the words around it, while Skip-gram predicts surrounding words from one word. GloVe is another embedding method that learns from how often words appear together in a large text collection.

Cosine similarity is used to measure how close two word vectors are. A higher score means the words are more related.

Static embeddings like Word2Vec and GloVe give each word one fixed vector. Contextual embeddings like BERT create different vectors depending on the sentence.

## Similarity Results

The script tested 10 sentiment-related words: good, bad, terrible, excellent, great, service, awful, happy, poor, and not.

Some useful results:

| Word      | Similar words                                     |
| --------- | ------------------------------------------------- |
| good      | better, really, always, sure, something           |
| bad       | worse, unfortunately, too, really, little         |
| terrible  | horrible, awful, tragic, dreadful, tragedy        |
| excellent | quality, good, skill, skills, superb              |
| awful     | horrible, terrible, dreadful, unbelievable, scary |
| not       | be, because, if, should, but                      |

## Analogy Results

| Analogy                     | Result  | Makes sense? |
| --------------------------- | ------- | ------------ |
| good → better :: bad → ?    | worse   | Yes          |
| good → best :: bad → ?      | ever    | No           |
| happy → happiest :: sad → ? | saddest | Yes          |
| man → king :: woman → ?     | queen   | Yes          |

Most analogy results made sense, especially “good → better :: bad → worse” and “man → king :: woman → queen”. However, one result failed because “good → best :: bad → ?” gave “ever” instead of “worst”.

## Intuitive Findings

The first intuitive finding is that negative words were close together. For example, “terrible”, “awful”, “horrible”, and “dreadful” appeared near each other. This makes sense because they are used in similar negative contexts.

The second intuitive finding is that some analogy relationships worked well. For example, the model returned “queen” for “man → king :: woman → ?”. This shows that embeddings can capture some relationships between words, not only simple similarity.

## Surprising Finding

One surprising result was that “bad” and “good” had high similarity. This seems strange because they are opposites. However, it happens because both words appear in similar sentence structures, such as “the service was good” and “the service was bad”. This shows that embeddings sometimes capture context similarity, not true meaning.

## Baseline Comparison

| Model                               | Accuracy |
| ----------------------------------- | -------- |
| TF-IDF + Logistic Regression        | 65.7%    |
| GloVe average + Logistic Regression | 74.3%    |

The averaged GloVe vectors performed better overall. This may be because pretrained embeddings already contain meaning learned from a large dataset. However, this does not mean negation is solved.

## Reflection

TF-IDF treats words like “excellent” and “great” as completely separate features. Word embeddings improve this because they place similar words close together in vector space. This helps the model understand that different positive words can have related meanings. For sentiment analysis, this can help when the dataset is small or when some words appear only a few times.

However, embeddings do not fully solve negation. For example, “good” is still a positive word, but “not good” is negative. If we average word vectors, the model may lose the order of the words and may not understand that “not” changes the meaning. So embeddings help with word meaning and similarity, but they still struggle with negation, sarcasm, and sentence context.
