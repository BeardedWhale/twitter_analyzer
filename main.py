import locale
from pprint import pprint
from typing import Optional, List, Dict, Any

import pytz
import twitter
import datetime

import constants

api = None

class TwitterApi():
    def get_api_instance(self):
        global api
        if api:
            return api
        api = twitter.Api(consumer_key=constants.consumer_id_key,
                          consumer_secret=constants.consumer_secret_key,
                          access_token_key=constants.access_token_key,
                          access_token_secret=constants.access_token_secret)
        api.VerifyCredentials()
        return api

    def is_follower_of(self, api, screen_name_1: str, screen_name_2: str):
        """
        Is screen_name_1 subscriber of screen_name_2
        :param api:
        :param screen_name_1:
        :param screen_name_2:
        :return:
        """

        if not api:
            return -1
        try:
            followers = api.GetFollowers(screen_name=screen_name_2, count=1)

            print(followers)
            print(screen_name_2)
            return screen_name_1 in followers
        except Exception as e:
            print(f'Exception occured: {e}')
        return -1


if __name__ == '__main__':
    twitterSearch = TwitterApi()
    api = twitterSearch.get_api_instance()
    is_follower = twitterSearch.is_follower_of(api=api, screen_name_1='NASA', screen_name_2='SpaceX')
    print(is_follower)
