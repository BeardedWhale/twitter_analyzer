import datetime
import json
import os
import time
from typing import List, Dict


import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from nlk.corpus import stopwords
from nltk.tokenize import word_tokenize

from dataset.constants import RETWEETED_STATUS_KEY, USER_KEY, SCREEN_NAME_KEY, USER_MENTIONS_KEY, \
    IN_REPLY_TO_SCREEN_NAME_KEY, RETWEETS_KEY, MENTIONS_KEY, COMMENTS_KEY, DESCRIPTION_SIMILARITY, FOLLOWING_SIMILARITY, \
    HASHTAGS_SIMILARITY, CATEGORIES_SIMILARITY, INTERACTION_VECTOR_KEY, \
    SIMILARITY_VECTOR_KEY, NUMBER_OF_COMMENTS_KEY, LIST_OF_COMMENTS_KEY, COMMENTS_INTERACTION, RETWEETS_INTERACTION, \
    LIST_OF_RETWEETS_KEY, LIST_OF_MENTIONS_KEY, LIST_OF_LIKES_KEY, LIST_OF_FOLLOWING_KEY, FOLLOWING_INTERACTION, \
    LIKES_INTERACTION, DESCRIPTION_KEY, LIST_OF_HASHTAGS_KEY, LIST_OF_CATEGORIES_KEY, CONSUMER_ID_KEY, \
    CONSUMER_SECRET_KEY, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, MENTIONS_INTERACTION


class DocSim:
    """
    Calculate similarity between documents
    """

    def __init__(self):
        try:
            model_path = os.environ['WORD2VEC_VECTORS']
            print(datetime.datetime.now())
            self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            print(datetime.datetime.now())
            self.w2v_model.init_sims(replace=True)  # normalize vectors
            print(datetime.datetime.now())


        except KeyError as e:
            print( f'Word2vec vectors were not found. Error: {e}')

        self.stopwords = list(set(stopwords.words('english')))

    def vectorise(self, doc: str) -> np.array:
        """
        Identify the vector values for each word in the given document
        :param doc: a document to vectorise
        :return: a mean vector of word2vec vectors of the document
        """
        doc = doc.lower()
        words = [w for w in doc.split(' ') if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vec_a: np.array, vec_b: np.array):
        """
        Find the cosine similarity distance between two vectors.
        :param vec_a: a vector
        :param vec_b: a vector
        :return: a cosine similarity
        """
        csim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_cosine_sim(self, source_vec, doc: str) -> float:
        """
        Calculates & returns similarity scores between given source document & target_document.
        :param source_doc: a source document
        :param doc: doc to compare with source_doc
        :return: float number that characterises similarity between two documents
        """
        target_vec = self.vectorise(doc)
        similarity = self._cosine_sim(source_vec, target_vec)

        return similarity

    def calculate_wmd_sim(self, source_doc: str, doc: str) -> float:
        """
        Calculates & returns negative world movers' distance between given source document & target_document.
        :param source_doc: a source document
        :param doc: doc to compare with source_doc
        :return: float number that characterises similarity between two documents
        """
        wmdistance = self.w2v_model.wmdistance(source_doc.split(), doc.split())

        return 1. / (1. + wmdistance)  # similarity is a negative distance

    def calculate_similarity(self, source_doc: str, target_docs: List[str], threshold=0, cos_similarity_weight=0.5) -> \
            List[Dict]:
        """
        Calculates similarity of documents using 2 methods (calculate_cosine_sim and calculate_wmd_sim).
        Returns weighted sum of these two metrics as a sim score between source_doc and all target_docs.
        :param source_doc: a source document
        :param target_doc: a list of documents to check
        :param threshold: a min threshold of similarity
        :param cos_similarity_weight: a weight of cosine similarity,
         wmd similarity is multiplied by (1-cos_similarity_weight)
        :return: a sorted list of docs with similarity score
        """
        if not 0 <= cos_similarity_weight <= 1:
            cos_similarity_weight = 0.5
        start = time.time()
        source_vec = self.vectorise(source_doc)

        results = []
        for doc in target_docs:
            start = time.time()
            wmdsimilarity = self.calculate_wmd_sim(source_doc=source_doc, doc=doc)
            cosine_similarity = self.calculate_cosine_sim(source_vec=source_vec, doc=doc)
            # calculating weighted sum as general similarity between documents
            sim_score = cos_similarity_weight * (cosine_similarity - wmdsimilarity) + wmdsimilarity

            results.append({
                'score': sim_score,
                'doc': doc
            })

        results.sort(key=lambda k: k['score'], reverse=True)
        return results



doc_sim = DocSim()

tramp = ["President of the United States", "United States Senate", "Democratic Party",
         "United States", "Election Day", "White House", "California",
         "United States House of Representatives", "United States Congress", "Claire McCaskill",
         "Republican Party", "Border", "Vice President of the United States",
         "Political party strength in Florida", "George W. Bush"]
hillary = ["Elections", "Voter turnout", "Democracy", "Voter registration",
           "Voting", "Election", "Voting system", "Minnesota", "Voter suppression",
           "Electoral roll", "Electronic voting", "Voting machine", "Poll", "Voting Rights Act",
           "Help America Vote Act"]

users_data = ''
user_pairs_data = ''
with open('/Users/elizavetabatanina/PycharmProjects/twitter_analyzer/twitter_analyzer/dataset/user_whole_data.json') as file:
    users_data = file.read()
    users_data = json.loads(users_data)
with open('/Users/elizavetabatanina/PycharmProjects/twitter_analyzer/twitter_analyzer/dataset/pairs_data copy.txt') as file:
    user_pairs_data = file.read()
    user_pairs_data = json.loads(user_pairs_data)

for user in users_data:
    for user_2 in users_data:
        categories_1 = user[USER_KEY][LIST_OF_CATEGORIES_KEY]
        categories_2 = user_2[USER_KEY][LIST_OF_CATEGORIES_KEY]
        hashtags_1 = user[USER_KEY][LIST_OF_HASHTAGS_KEY]
        hashtags_2 = user_2[USER_KEY][LIST_OF_HASHTAGS_KEY]
        description_1 = user_2[USER_KEY][DESCRIPTION_KEY]
        description_2 = user_2[USER_KEY][DESCRIPTION_KEY]
        user_screen_name = user[USER_KEY][SCREEN_NAME_KEY]
        user_2_screen_name = user_2[USER_KEY][SCREEN_NAME_KEY]
        user_2_vector = user_pairs_data[user_screen_name.replace('\n','')]['users'].get(
            user_2_screen_name.replace('\n', ''), {})
        if user_2_vector:
            user_2_vector[SIMILARITY_VECTOR_KEY][CATEGORIES_SIMILARITY] = \
                doc_sim.calculate_similarity(''.join(categories_1), [' '.join(categories_2)], cos_similarity_weight=1.0)[0]['score']

            hashtags_similarity = doc_sim.calculate_similarity(''.join(hashtags_1),
                                                               [' '.join(hashtags_2)], cos_similarity_weight=1.0)[0]['score']
            if hashtags_similarity == 0:
                hashtags_1_set = set(hashtags_1)
                hashtags_2_set = set(hashtags_2)

                common_hashtags = list(hashtags_1_set.intersection(hashtags_2_set))
                if len(hashtags_1) == 0:
                    hashtags_similarity = 0
                else:
                    hashtags_similarity = len(common_hashtags) / len(hashtags_1_set)

            user_2_vector[SIMILARITY_VECTOR_KEY][HASHTAGS_SIMILARITY] = hashtags_similarity

            user_2_vector[SIMILARITY_VECTOR_KEY][DESCRIPTION_SIMILARITY] = \
                doc_sim.calculate_similarity(description_1, [description_2], cos_similarity_weight=1.0)[0]['score']
        else:
            print(user_screen_name, user_2_screen_name)

print(datetime.datetime.now())
with open('/Users/elizavetabatanina/PycharmProjects/twitter_analyzer/twitter_analyzer/dataset/pairs_data copy v3.txt', 'w+') as file:
    user_pairs_data = json.dumps(user_pairs_data)
    file.write(user_pairs_data)

tramp = ''.join(tramp)
hillary = ''.join(hillary)

# print(doc_sim.calculate_similarity(tramp, [hillary]))
