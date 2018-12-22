import locale
import os
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



API_KEYS = [{CONSUMER_ID_KEY: os.environ[f'{CONSUMER_ID_KEY}_1'],
             CONSUMER_SECRET_KEY: os.environ[f'{CONSUMER_SECRET_KEY}_1'],
             ACCESS_TOKEN_KEY: os.environ[f'{ACCESS_TOKEN_KEY}_1'],
             ACCESS_TOKEN_SECRET:os.environ[f'{ACCESS_TOKEN_SECRET}_1']},
            {CONSUMER_ID_KEY: os.environ[f'{CONSUMER_ID_KEY}_2'],
             CONSUMER_SECRET_KEY: os.environ[f'{CONSUMER_SECRET_KEY}_2'],
             ACCESS_TOKEN_KEY: os.environ[f'{ACCESS_TOKEN_KEY}_2'],
             ACCESS_TOKEN_SECRET: os.environ[f'{ACCESS_TOKEN_SECRET}_2']},
            {CONSUMER_ID_KEY: os.environ[f'{CONSUMER_ID_KEY}_3'],
             CONSUMER_SECRET_KEY: os.environ[f'{CONSUMER_SECRET_KEY}_3'],
             ACCESS_TOKEN_KEY: os.environ[f'{ACCESS_TOKEN_KEY}_3'],
             ACCESS_TOKEN_SECRET: os.environ[f'{ACCESS_TOKEN_SECRET}_3']}]
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
        if not api:
            api = self.get_api_instance()
        keys_index = keys_index%len(API_KEYS)
        api.ClearCredentials()
        keys = API_KEYS[keys_index]
        api.SetCredentials(consumer_key=keys[CONSUMER_ID_KEY],
                           consumer_secret=keys[CONSUMER_SECRET_KEY],
                           access_token_key=keys[ACCESS_TOKEN_KEY],
                           access_token_secret=keys[ACCESS_TOKEN_SECRET])
        api.VerifyCredentials()
        return api


    def find_posts_twitter(self, screen_name: str, pool_amount: int, since: datetime,
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
        api = self.get_api_instance()
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


    def get_friends_of(self, screen_name)-> List:
        api = self.get_api_instance()
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


    def common_subscriptions_a_to_b(self,mscreen_name_a: str, screen_name_b: str):
        """
        If user 1 and user 2 are subscribed to one account, the name of the common acc. will be added to result list
        """
        api = self.get_api_instance()
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

    def get_favorites_users(self, screen_name, since: datetime,
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
        api = self.get_api_instance()
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
