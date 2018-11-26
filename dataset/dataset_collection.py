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



API_KEYS = [{CONSUMER_ID_KEY: 'meq7EApEAfRF8jxpGqr0qxqFX',
             CONSUMER_SECRET_KEY: 'wHPYg8feuE4zvzDRYRXbGeH9d1vaHcVv7Zw03MIon6Y4WbDjE0',
             ACCESS_TOKEN_KEY: '1058944938887409664-GDAlXxbFQYze8gcPDRKYwtebzDhRdq',
             ACCESS_TOKEN_SECRET: 'ttLrqiM0ejlJyEN7PFiE4CcIqWsuukg6KW7D0srtICFz9'},
            {CONSUMER_ID_KEY: 'pRs7lC697sJpY5FqmxS4Fpy2j',
             CONSUMER_SECRET_KEY: 'CjrPspD8t2Iz9Skk0WfW7pVi6VdwV7oStOn9XlnDzPIA7SWKr6',
             ACCESS_TOKEN_KEY: '1058944938887409664-2wzfwfnccNLFFs742j1Eh4TTVo98v0',
             ACCESS_TOKEN_SECRET: 'IRnR2K9ZDbA4Nt7WrlP4XsxMAj9zC3GCvW7FPNNpwmIMo'},
            {CONSUMER_ID_KEY: 'NC19WDNaMoEaaV9s8nadVUvBI',
             CONSUMER_SECRET_KEY: 'rtoYVT9AykdYQWWv5Nh7ZdDode72DRSLX8XswRrqprxhC2TnI3',
             ACCESS_TOKEN_KEY: '1058944938887409664-402KNxXTgNNhMoAK3pmAUziB3Fhxzj',
             ACCESS_TOKEN_SECRET: 'vVxGZQdN9ubMK1EnxZGx62ZsxVqwFQiwDZmXmEMfPYSKj'}]
api = None


class TwitterApi():

    current_keys_index = 0
    def get_api_instance(self):
        global api
        if api:
            return api
        keys = API_KEYS[0]
        api = twitter.Api(consumer_key=keys[CONSUMER_ID_KEY],
                          consumer_secret=keys[CONSUMER_SECRET_KEY],
                          access_token_key=keys[ACCESS_TOKEN_KEY],
                          access_token_secret=keys[ACCESS_TOKEN_SECRET])
        api.VerifyCredentials()
        return api

    def update_api_instance(self, keys_index: int = 0):
        """
        Updates api instance in case of rate limit
        :param keys_index:
        :return:
        """
        global api
        keys_index = keys_index%len(API_KEYS)
        api.ClearCredentials()
        keys = API_KEYS[keys_index]
        api.SetCredentials(consumer_key=keys[CONSUMER_ID_KEY],
                           consumer_secret=keys[CONSUMER_SECRET_KEY],
                           access_token_key=keys[ACCESS_TOKEN_KEY],
                           access_token_secret=keys[ACCESS_TOKEN_SECRET])
        api.VerifyCredentials()
        return api


    def find_posts_twitter(self, api, screen_name: str, pool_amount: int, since: datetime,
                           until: datetime = datetime.datetime.now(datetime.timezone.utc)) -> List[Dict[str, Any]]:
        """
        Finds user's posts and sorts them b amount of likes
        :param api: instance of twitter api
        :param screen_name: twitter screen name of a user
        :param pool_amount: amount of posts to retrieve (if 0 we retrieve posts by time period [since; until])
        :param since: datetime of oldest post (is used only if pool amount is not mentioned)
        :param until: datetime of latest post (is used only if pool amount is not mentioned)
        :return: list of post dictionaries
        """

        if not api:
            return []
        if not since:
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        posts = []
        if pool_amount is None or pool_amount == 0:
            pool_amount = 20

        try:
            if since and since > datetime.datetime.now(datetime.timezone.utc):
                since = datetime.datetime.now(datetime.timezone.utc)
            if until and until > datetime.datetime.now(datetime.timezone.utc):
                until = datetime.datetime.now(datetime.timezone.utc)

            if since and until and since > until:
                return []

            last_id = ''
            count = 0
            search_finish = False  # flag that indicates that search is finished
            first_cycle = True
            while (count < pool_amount or pool_amount == -1) and not search_finish:
                count_to_request = 200
                number_of_tries = 0
                fail = True
                time_line = []
                while fail:
                    try:
                        time_line = api.GetUserTimeline(screen_name=screen_name, count=count_to_request, max_id=last_id)
                        fail = False
                    except Exception as e:
                        api = self.update_api_instance(number_of_tries)
                        number_of_tries +=1
                        if number_of_tries > 10:
                            print(f'something bad happened. Tried 10 times, it doesn\'t help e: {e}')
                            break
                        if number_of_tries%3 == 0:
                            print("[TIMELINE] ALL KEYS LIMIT EXCEEDED. GOING TO SLEEP FOR 5 MINUTES")
                            time.sleep(300)

                if not time_line:
                    break

                # popping first element from timeline as it was received in previous query
                if not first_cycle:
                    time_line.pop(0)

                first_cycle = False

                # adding posts from timeline to posts that satisfy required timeperiod
                for status in time_line:
                    post = status.AsDict()
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        last_id = post['id_str']
                        date = post['created_at']
                        date_parsed = self.get_tweet_date(date)
                        post['created_at'] = self.serialize_datetime(date_parsed)
                        posts.append(post)
                        count += 1

                    elif posts:
                        last_id = post['id_str']
                        search_finish = True
                        break

            sorted_posts = sorted(posts, key=lambda x: int(x.get('favorite_count', 0)), reverse=True)
            if pool_amount != -1:
                result = sorted_posts[:pool_amount]
            else:
                result = sorted_posts
        except Exception as e:
            print(f'Exception occured: {e}')
            return []

        return result

    def find_retweets_twitter(self, posts: List[Dict], screen_name: str) -> List[Dict[str, Any]]:
        """
        Finds retweets from given user's screen_namelist in given list of posts
        :param posts: list of posts in some user timeline
        :param screen_name: user that was an author of tweet that was retweeted
        :return: list of retweet dictionaries
        """
        retweets = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                if author_screen_name == screen_name:
                    retweets.append(post)
        return retweets

    def get_posts_information(self, posts: List[Dict]) -> Tuple[List, List, List]:
        """
        Gets a list of all users that were mentioned, commented and retweeted in given posts
        :param posts:
        :return:
        """
        retweeted_users = []
        mentioned_users = []
        commented_users = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                retweeted_users.append(author_screen_name)
            if USER_MENTIONS_KEY in post:
                if IN_REPLY_TO_SCREEN_NAME_KEY in post:
                    reply_screen_name = post.get(IN_REPLY_TO_SCREEN_NAME_KEY, '')
                    commented_users.append(reply_screen_name)
                else:
                    users_mentioned = post.get(USER_MENTIONS_KEY, [])
                    for user in users_mentioned:
                        user_screen_name = user.get(SCREEN_NAME_KEY, '')
                        commented_users.append(user_screen_name)
        return retweeted_users, mentioned_users, commented_users

    def find_user_interactions_in_posts(self, posts: List[Dict], screen_name: str) -> Dict[str, Any]:
        """

        :param posts:
        :param screen_name:
        :return:
        """
        interactions = {}
        retweets = []
        mentions = []
        comments = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                if author_screen_name == screen_name:
                    retweets.append(post)
                    continue  # not to add retweets to mentions
            if USER_MENTIONS_KEY in post:
                if IN_REPLY_TO_SCREEN_NAME_KEY in post:
                    reply_screen_name = post.get(IN_REPLY_TO_SCREEN_NAME_KEY, '')
                    if reply_screen_name == screen_name:
                        comments.append(post)
                else:
                    users_mentioned = post.get(USER_MENTIONS_KEY, [])
                    for user in users_mentioned:
                        user_screen_name = user.get(SCREEN_NAME_KEY, '')
                        if user_screen_name == screen_name:
                            mentions.append(post)

        return {RETWEETS_KEY: retweets,
                MENTIONS_KEY: mentions,
                COMMENTS_KEY: comments}

    def get_tweet_date(self, date: str):
        try:
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
            return datetime.datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y')
        except Exception as e:
            print(f'error has occured: {e}')

    def serialize_datetime(self, date: datetime) -> str:
        """
        Make a string representing a datetime in UTC. The resulting string
        can be deserialized by deserialize_datetime().

        If the datetime given is naive or is not in UTC, then exception is raised.

        Milliseconds can be lost in serialization.

        :param date: UTC datetime to serialize
        :return: datetime string
        """
        if date.tzinfo not in [pytz.UTC, datetime.timezone.utc]:
            raise ValueError('Datetime should not be naive and should be in UTC (pytz.UTC or datetime.timezone.utc)')
        return date.strftime('%Y-%m-%dT%H:%M:%S%z')


    def get_friends_of(self, api, screen_name)-> List:
        if not api:
            return []
        try:
            number_of_tries = 0
            fail = True
            friends = []
            while fail:
                try:
                    friends = api.GetFriends(screen_name=screen_name)
                    fail = False
                except Exception as e:
                    api = self.update_api_instance(number_of_tries)
                    number_of_tries += 1
                    if number_of_tries > 10:
                        print(f'something bad happend. Tried 10 times, it doesn\'t help e: {e}')
                        break

                    if number_of_tries % 3 == 0:
                        print(f"[FRIENDS] ALL KEYS LIMIT EXCEEDED. GOING TO SLEEP FOR 5 MINUTES. SCREEN_NAME: {screen_name}")
                        time.sleep(300)

            followers_screen_names = [friend.screen_name for friend in friends]

            return followers_screen_names

        except Exception as e:
            print(f'Exception occured: {e}')
        return []

    # S
    def common_subscriptions_a_to_b(self, api, screen_name_a: str, screen_name_b: str):
        """
        If user 1 and user 2 are subscribed to one account, the name of the common acc. will be added to result list
        """
        if not api:
            return -1
        try:
            friends_1 = api.GetFriends(screen_name=screen_name_a)
            friends_2 = api.GetFriends(screen_name=screen_name_b)

            common_list = []

            for friend_1 in friends_1:
                for friend_2 in friends_2:
                    if friend_2.screen_name == friend_1.screen_name:
                        common_list.append(friend_1.screen_name)

            return len(common_list) / len(friends_1)
        except Exception as e:
            print(f'Exception occured: {e}')

        return -1

    def get_favorites_users(self, api, screen_name, since: datetime,
                            until: datetime = datetime.datetime.now(datetime.timezone.utc)) -> List[str]:
        """
        Returns list of people whos posts were liked
        :param api:
        :param screen_name:
        :param screen_name2:
        :param since:
        :param until:
        :return: integer number indicating how many times screen_name liked screen name2
                -1 on error
        """
        if not api:
            return []

        try:
            since, until = self.parse_time_interval(since, until)
            if since and until and since > until:
                return []

            search_done = False
            last_id = ''
            count = 0
            favorites_authors = []
            while not search_done:
                number_of_tries = 0
                fail = True
                favorites = []

                while fail:
                    try:
                        favorites = api.GetFavorites(screen_name=screen_name, count=200, since_id=last_id)
                        fail = False
                    except Exception as e:
                        api = self.update_api_instance(number_of_tries)
                        number_of_tries += 1
                        if number_of_tries > 10:
                            print(f'something bad happened. Tried 10 times, it doesn\'t help e: {e}')
                            break
                        if number_of_tries % 3 == 0:
                            print(f"[FAVORITES] ALL KEYS LIMIT EXCEEDED. GOING TO SLEEP FOR 5 MINUTES. SCREEN_NAME: {screen_name}")
                            time.sleep(300)

                for fav in favorites:
                    post = fav.AsDict()
                    last_id = post['id_str']
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        post_author = post['user']['screen_name']
                        favorites_authors.append(post_author)
                    elif since > self.get_tweet_date(post['created_at']):
                        search_done = True
                        break

            return favorites_authors
        except Exception as e:
            print(f'Exception occured: {e}')
            return []

    def parse_time_interval(self, since, until: datetime):
        if not since:
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

        if since and since > datetime.datetime.now(datetime.timezone.utc):
            since = datetime.datetime.now(datetime.timezone.utc)
        if until and until > datetime.datetime.now(datetime.timezone.utc):
            until = datetime.datetime.now(datetime.timezone.utc)

        return since, until

    # S
    def get_hashtags_similarity(self, posts1, posts2):
        """
        Calculates similarity of posts of two users based on hashtags
        :param posts1: list of posts of 1st user
        :param posts2: list of posts of user to compare with
        :return: percentage(number > 0 and < 1) of common hashtags w.r.t the 1st user
        """
        tags1 = []
        for post in posts1:
            for tag in post['hashtags']:
                tags1.append(tag)
        tags2 = []
        for post in posts2:
            for tag in post['hashtags']:
                tags2.append(tag)

        if not len(tags1) or not len(tags2):
            return 0
        count = 0
        for tag1 in tags1:
            for tag2 in tags2:
                if tag1 == tag2:
                    count += 1
        return count / len(tags1)

    def _get_natural_language_understanding(self, version):
        n = NaturalLanguageUnderstandingV1(
            version=version,
            iam_apikey='SGjJgUAGXQEdbXiRe27u2V4hmeMIrEESo0vcXrfCunLL',
            url='https://gateway-wdc.watsonplatform.net/natural-language-understanding/api'
        )
        return n

    def _parse_posts_to_text(self, posts):
        text = []
        for post in posts:
            if post['lang'] == 'en':
                text.append(post['text'].replace('\n', ''))
        return ''.join(text)

    def get_categories(self, posts):
        """
        :param posts: list of posts to analyze
        :return: list of unique categories
        """
        text = self._parse_posts_to_text(posts)
        if not len(text):
            return []
        natural_language_understanding = self._get_natural_language_understanding('2018-03-16')
        categories = []

        try:
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(concepts=ConceptsOptions(limit=20))).get_result()
            for category in response['concepts']:
                c = category['text']
                if c == '' or c in categories:
                    continue
                categories.append(c)
        except Exception:
            return []
        return categories

    # S
    def get_categories_similarity(self, categories1, categories2):
        """
         Calculates users' similarity of posts based on categories from their posts
        :param posts1: list of posts of the 1st user
        :param posts2: list of posts of the user to compare with
        :return: percentage(number > 0 and < 1) of common categories w.r.t to the 1st user
        """

        if not len(categories1) or not len(categories2):
            return 0
        count = 0
        for category1 in categories1:
            for category2 in categories2:
                if category1 == category2:
                    count += 1
        return count / len(categories1)

    def get_all_hashtags(self, posts):
        hashtags = []
        for post in posts:
            hashtags.extend(hashtag['text'] for hashtag in post['hashtags'])

        return hashtags


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
        interaction_vector[COMMENTS_INTERACTION] = user_1[LIST_OF_COMMENTS_KEY].count(user_2_screen_name)
        interaction_vector[RETWEETS_INTERACTION] = user_1[LIST_OF_RETWEETS_KEY].count(user_2_screen_name)
        interaction_vector[MENTIONS_INTERACTION] = user_1[LIST_OF_MENTIONS_KEY].count(user_2_screen_name)
        interaction_vector[FOLLOWING_INTERACTION] = user_1[LIST_OF_FOLLOWING_KEY].count(user_2_screen_name)
        interaction_vector[LIKES_INTERACTION] = user_1[LIST_OF_LIKES_KEY].count(user_2_screen_name)
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
    file_name = "user_data.json"
    dataset.save_posts_of_user(twitter_search=twitterSearch,
                               screen_names=dataset.read_users('accounts.txt'),
                               file_name=file_name)
    users_pair_info = dataset.get_user_pairs_info_from_file(file_name)
    data_to_save = json.dumps(users_pair_info)

    with open('pairs_data.txt', 'w+') as f:
        f.write(data_to_save)

