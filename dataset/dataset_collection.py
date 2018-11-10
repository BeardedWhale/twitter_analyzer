import locale
from pprint import pprint
from typing import Optional, List, Dict, Any

import pytz
import twitter
import datetime

from dataset.constants import RETWEETED_STATUS_KEY, USER_KEY, SCREEN_NAME_KEY, USER_MENTIONS_KEY, \
    IN_REPLY_TO_SCREEN_NAME_KEY, RETWEETS_KEY, MENTIONS_KEY, COMMENTS_KEY

consumer_id_key = 'NC19WDNaMoEaaV9s8nadVUvBI'
consumer_secret_key = 'rtoYVT9AykdYQWWv5Nh7ZdDode72DRSLX8XswRrqprxhC2TnI3'
access_token_key = '1058944938887409664-402KNxXTgNNhMoAK3pmAUziB3Fhxzj'
access_token_secret = 'vVxGZQdN9ubMK1EnxZGx62ZsxVqwFQiwDZmXmEMfPYSKj'
api = None

class TwitterApi():


    def get_api_instance(self):
        global api
        if api:
            return api
        api = twitter.Api(consumer_key=consumer_id_key,
                              consumer_secret=consumer_secret_key,
                              access_token_key=access_token_key,
                              access_token_secret=access_token_secret)
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
            pool_amount =20

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
                time_line = api.GetUserTimeline(screen_name=screen_name, count=count_to_request, max_id=last_id)
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

    def find_retweets_twitter(self, posts: List[Dict], screen_name: str)->List[Dict[str, Any]]:
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

    def find_mentions_twitter(self, posts: List[Dict], screen_name: str)->List[Dict[str, Any]]:
        """

        :param posts:
        :param screen_name:
        :return:
        """
    def find_user_interactions_in_posts(self, posts: List[Dict], screen_name: str)-> Dict[str, Any]:
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
                    continue # not to add retweets to mentions
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

if __name__ == '__main__':
    twitterSearch = TwitterApi()
    posts = twitterSearch.find_posts_twitter(api=twitterSearch.get_api_instance(), screen_name='', pool_amount=50, since=None)
    # pprint(posts)
    interactions = twitterSearch.find_user_interactions_in_posts(posts, '')
    pprint(interactions)
import locale
from typing import Optional, List, Dict, Any

import pytz
import twitter
import datetime

consumer_id_key = 'NC19WDNaMoEaaV9s8nadVUvBI'
consumer_secret_key = 'rtoYVT9AykdYQWWv5Nh7ZdDode72DRSLX8XswRrqprxhC2TnI3'
access_token_key = '1058944938887409664-402KNxXTgNNhMoAK3pmAUziB3Fhxzj'
access_token_secret = 'vVxGZQdN9ubMK1EnxZGx62ZsxVqwFQiwDZmXmEMfPYSKj'
api = None


class TwitterApi():

    def get_api_instance(self):
        global api
        if api:
            return api
        api = twitter.Api(consumer_key=consumer_id_key,
                          consumer_secret=consumer_secret_key,
                          access_token_key=access_token_key,
                          access_token_secret=access_token_secret)
        api.VerifyCredentials()
        return api

    def find_posts_twitter(self, api, screen_name: str, pool_amount: int, since: datetime,
                           until: datetime = datetime.datetime.now(datetime.timezone.utc)) -> List[Dict[str, Any]]:

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
                time_line = api.GetUserTimeline(screen_name=screen_name, count=count_to_request, max_id=last_id)
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

    # calculates how many times user with screen_name liked user with screen_name2
    def get_favorites_count(self, api, screen_name, screen_name2: str, since: datetime,
                            until: datetime = datetime.datetime.now(datetime.timezone.utc)):
        if not api:
            return -1

        try:
            since, until = twitterSearch.parse_time_interval(since, until)
            if since and until and since > until:
                return -1

            search_done = False
            last_id = ''
            count = 0
            while not search_done:
                favorites = api.GetFavorites(screen_name=screen_name, count=200, since_id=last_id)
                for fav in favorites:
                    post = fav.AsDict()
                    print(post['user']['screen_name'])
                    last_id = post['id_str']
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        if post['user']['screen_name'] == screen_name2:
                            count += 1
                    elif since > self.get_tweet_date(post['created_at']):
                        search_done = True
                        break

            return count
        except Exception as e:
            print(f'Exception occured: {e}')
            return -1

    def parse_time_interval(self, since, until: datetime):
        if not since:
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

        if since and since > datetime.datetime.now(datetime.timezone.utc):
            since = datetime.datetime.now(datetime.timezone.utc)
        if until and until > datetime.datetime.now(datetime.timezone.utc):
            until = datetime.datetime.now(datetime.timezone.utc)

        return since, until


if __name__ == '__main__':
    twitterSearch = TwitterApi()
    posts = twitterSearch.find_posts_twitter(api=twitterSearch.get_api_instance(), screen_name='BeardedRain',
                                             pool_amount=20, since=None)
    print(posts)
    count = twitterSearch.get_favorites_count(api=twitterSearch.get_api_instance(), screen_name='BeardedRain',
                                              screen_name2='elonmusk', since=None)
    print(count)
