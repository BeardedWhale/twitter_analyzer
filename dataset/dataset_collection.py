import locale
from pprint import pprint
import time
from typing import Optional, List, Dict, Any, Tuple
from typing import List, Dict, Any
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions, \
    ConceptsOptions

import pytz
import twitter
import datetime
import json

from dataset.constants import RETWEETED_STATUS_KEY, USER_KEY, SCREEN_NAME_KEY, USER_MENTIONS_KEY, \
    IN_REPLY_TO_SCREEN_NAME_KEY, RETWEETS_KEY, MENTIONS_KEY, COMMENTS_KEY, DESCRIPTION_SIMILARITY, FOLLOWING_SIMILARITY, \
    HASHTAGS_SIMILARITY, CATEGORIES_SIMILARITY, INTERACTION_VECTOR_KEY, \
    SIMILARITY_VECTOR_KEY, NUMBER_OF_COMMENTS_KEY, LIST_OF_COMMENTS_KEY, COMMENTS_INTERACTION, RETWEETS_INTERACTION, \
    LIST_OF_RETWEETS_KEY, LIST_OF_MENTIONS_KEY, LIST_OF_LIKES_KEY, LIST_OF_FOLLOWING_KEY, FOLLOWING_INTERACTION, \
    LIKES_INTERACTION, DESCRIPTION_KEY, LIST_OF_HASHTAGS_KEY, LIST_OF_CATEGORIES_KEY, CONSUMER_ID_KEY, \
    CONSUMER_SECRET_KEY, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, MENTIONS_INTERACTION
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset.twitter_api import TwitterApi


class DatasetCollection():


    def save_posts_of_user(self, twitter_search: TwitterApi, screen_names, file_name):
        """
        Gets users info and posts, analyses this information and saves it to file
        :param twitter_search: instance of TwitterApi
        :param screen_names: names of users
        :param file_name: file to save posts
        :return: nothing
        """
        all_users = []
        for screen_name in screen_names:

            posts = twitter_search.find_posts_twitter(api=twitter_search.get_api_instance(), screen_name=screen_name,
                                                      pool_amount=-1,
                                                      since=datetime.datetime.now(
                                                         datetime.timezone.utc) - datetime.timedelta(31),
                                                      until=datetime.datetime.now(
                                                         datetime.timezone.utc))

            retweeted_users, mentioned_users, commented_users = twitter_search.get_posts_information(posts)
            favorite_users = twitter_search.get_favorites_users(twitterSearch.get_api_instance(), screen_name,
                                                                since=datetime.datetime.now(datetime.timezone.utc)
                                                                - datetime.timedelta(90),
                                                                until=datetime.datetime.now(datetime.timezone.utc))
            print(screen_name + "\n")
            all_posts_with_user = {}
            user = {}

            if len(posts) > 0:
                user[SCREEN_NAME_KEY] = screen_name
                user[LIST_OF_COMMENTS_KEY] = commented_users  # лист кого комментировал
                user[LIST_OF_RETWEETS_KEY] = retweeted_users  # лист кого ретвител
                user[LIST_OF_MENTIONS_KEY] = mentioned_users  # лист кого упоминал
                user[LIST_OF_LIKES_KEY] = favorite_users  # лист кого он лайкал
                user[LIST_OF_FOLLOWING_KEY] = twitter_search.get_friends_of(api=twitterSearch.get_api_instance(),
                                                                            screen_name=screen_name)  # кого он фолловит
                user[DESCRIPTION_KEY] = posts[0][USER_KEY][DESCRIPTION_KEY]  # описание
                user[LIST_OF_HASHTAGS_KEY] = twitter_search.get_all_hashtags(posts)  # хештег лист
                user[LIST_OF_CATEGORIES_KEY] = twitterSearch.get_categories(posts)  # лист категорий
            else:
                user_req = api.GetUser(screen_name=screen_name)
                user[SCREEN_NAME_KEY] = screen_name
                user[LIST_OF_COMMENTS_KEY] = []  # лист кого комментировал
                user[LIST_OF_RETWEETS_KEY] = []  # лист кого ретвител
                user[LIST_OF_MENTIONS_KEY] = []  # лист кого упоминал
                user[LIST_OF_LIKES_KEY] = favorite_users  # лист кого он лайкал
                user[LIST_OF_FOLLOWING_KEY] = twitter_search.get_friends_of(screen_name=screen_name)  # кого он фолловит
                user[DESCRIPTION_KEY] = user_req.description  # описание
                user[LIST_OF_HASHTAGS_KEY] = []  # хештег лист
                user[LIST_OF_CATEGORIES_KEY] = []  # лист категорий

            all_posts_with_user[USER_KEY] = user

            all_users.append(all_posts_with_user)

        json_post = json.dumps(all_users)

        with open(file_name, 'w+') as file:
            file.write(json_post)

    def read_users(self, file_name: str) -> List[str]:
        """
        Reads user names to analyze
        :param file_name: file with user names
        :return:
        """
        result_users = []
        with open(file_name) as file:
            for line in file:
                result_users.append(line)
        return result_users

    def _get_interaction_vector(self, user_1, user_2)-> Dict:
        """
        Computes interaction vector
        :param user_1:
        :param user_2:
        :return:
        """
        user_1 = user_1[USER_KEY]
        user_2_screen_name = user_2[USER_KEY][SCREEN_NAME_KEY]
        interaction_vector = {}
        interaction_vector[COMMENTS_INTERACTION] = user_1[LIST_OF_COMMENTS_KEY].count(user_2_screen_name.replace('\n',''))
        interaction_vector[RETWEETS_INTERACTION] = user_1[LIST_OF_RETWEETS_KEY].count(user_2_screen_name.replace('\n', ''))
        interaction_vector[MENTIONS_INTERACTION] = user_1[LIST_OF_MENTIONS_KEY].count(user_2_screen_name.replace('\n', ''))
        interaction_vector[FOLLOWING_INTERACTION] = user_1[LIST_OF_FOLLOWING_KEY].count(user_2_screen_name.replace('\n', ''))
        interaction_vector[LIKES_INTERACTION] = user_1[LIST_OF_LIKES_KEY].count(user_2_screen_name.replace('\n', ''))
        return interaction_vector

    def _get_auxilirary_vector(self, user):
        """

        :param user: user for whom to return dictionary
        :return: {COMMENTS: int, RETWEETS: int, MENTIONS:int, LIKES:int, FOLLOWINGS:int}

        """
        auxil = {}
        user = user[USER_KEY]
        # comments  [in timestamp]
        auxil['COMMENTS'] = len(user[LIST_OF_COMMENTS_KEY])
        # retweets [in timestamp]
        auxil['RETWEETS'] = len(user[LIST_OF_RETWEETS_KEY])
        # mentions [in timestamp]
        auxil['MENTIONS'] = len(user[LIST_OF_MENTIONS_KEY])
        # likes [in timestamp]
        auxil['LIKES'] = len(user[LIST_OF_LIKES_KEY])
        # following [all]
        auxil['FOLLOWINGS'] = len(user[LIST_OF_FOLLOWING_KEY])

        return auxil

    def get_pair_user_vectors(self, users: List[Dict])-> Dict[str, Any]:
        """
        calculates interactions, similarity and auxiliary variables for pair of users
        to calculate relationship strength
        :param users:
        :return:
        """
        all_user_pairs = {}
        for i, user_1 in enumerate(users):
            user_screen_name = user_1[USER_KEY][SCREEN_NAME_KEY]
            user_screen_name = user_screen_name.replace('\n', '')
            user_pairs = {} # consists of user + vector of interaction and similarity
            user_auxiliary_vars = self._get_auxilirary_vector(user_1)
            for j, user_2 in enumerate(users):
                if i != j:
                    interaction_vector = self._get_interaction_vector(user_1=user_1, user_2=user_2)
                    similarity_vector = self.get_user_simlarity_vector(user_1, user_2)
                    user_2_screen_name = user_2[USER_KEY][SCREEN_NAME_KEY]
                    user_2_screen_name = user_2_screen_name.replace('\n', '')
                    user_pairs[user_2_screen_name] = {INTERACTION_VECTOR_KEY: interaction_vector,
                                                      SIMILARITY_VECTOR_KEY: similarity_vector}
            user_info = {'auxiliary_vector': user_auxiliary_vars, 'users': user_pairs}
            all_user_pairs[user_screen_name] = user_info

        return all_user_pairs

    def get_user_simlarity_vector(self, user_1: Dict, user_2: Dict) -> Dict:
        """
        computes similarity vector for pair of users
        :param user_1:
        :param user_2:
        :return: dictionary with all similarity parameters
        """
        similarity = {}
        user_1 = user_1[USER_KEY]
        user_2 = user_2[USER_KEY]
        user_1_following = set(user_1[LIST_OF_FOLLOWING_KEY])
        user_2_following = set(user_2[LIST_OF_FOLLOWING_KEY])

        common_followings = list(user_1_following.intersection(user_2_following))
        following_sim = 1.0 * len(common_followings) / len(user_1_following)
        description_1 = user_1[DESCRIPTION_KEY]
        description_2 = user_2[DESCRIPTION_KEY]
        description_simlarity = self._calculate_description_similarity(description_1, description_2)

        hashtags_1 = set(user_1[LIST_OF_HASHTAGS_KEY])
        hashtags_2 = set(user_2[LIST_OF_HASHTAGS_KEY])
        common_hashtags = list(hashtags_1.intersection(hashtags_2))
        if len(hashtags_1) == 0:
            hashtags_similarity = 0
        else:
            hashtags_similarity = len(common_hashtags)/len(hashtags_1)
        categories_1 = user_1[LIST_OF_CATEGORIES_KEY]
        categories_2 = user_2[LIST_OF_CATEGORIES_KEY]
        categories_similarity = twitterSearch.get_categories_similarity(categories_1, categories_2)

        similarity[FOLLOWING_SIMILARITY] = following_sim
        similarity[DESCRIPTION_SIMILARITY] = description_simlarity
        similarity[HASHTAGS_SIMILARITY] = hashtags_similarity
        similarity[CATEGORIES_SIMILARITY] = categories_similarity

        return similarity


    def _calculate_description_similarity(self, description1, description2) -> float:
        """
        Computes cosine similarity of two descriptions of two users
        :param description1: descruption of user 1
        :param description2: description of user 2
        :return: number in range [0,1] that characterizes how users descriptions are similar
        """

        def get_vectors(*strs):
            text = [t for t in strs]
            vectorizer = CountVectorizer(text)
            vectorizer.fit(text)
            return vectorizer.transform(text).toarray()

        vectors = [t for t in get_vectors(description1, description2)]
        similarity = cosine_similarity(vectors)[0, 1]
        return similarity

    def read_users_info(self, file_name: str)-> List:
        """
        reads information about all user,
        i.e. user account info and posts hashtags and categories
        :param file_name: file with users info
        :return: dictionary with all data
        """
        file_content = ''
        with open(file_name) as f:
            file_content = f.read()

        users_info = json.loads(file_content)
        return users_info

    def get_user_pairs_info_from_file(self, file_name: str) -> Dict:
        """
        computes all nessesary vectors for user pairs analysis and saves to file
        :param file_name: name of file
        :return:
        """
        users_data = self.read_users_info(file_name)
        users_pair_info = self.get_pair_user_vectors(users_data)

        return users_pair_info



if __name__ == '__main__':
    twitterSearch = TwitterApi()
    dataset = DatasetCollection()
    posts = []
    file_name = "user_whole_data.json"
    users_pair_info = dataset.get_user_pairs_info_from_file(file_name)
    data_to_save = json.dumps(users_pair_info)

    with open('pairs_data.txt', 'w+') as f:
        f.write(data_to_save)

