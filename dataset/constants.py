import datetime

RETWEETED_STATUS_KEY = 'retweeted_status'
USER_KEY = 'user'
SCREEN_NAME_KEY = 'screen_name'
USER_MENTIONS_KEY = 'user_mentions'
IN_REPLY_TO_SCREEN_NAME_KEY = 'in_reply_to_screen_name'
RETWEETS_KEY = 'retweets'
COMMENTS_KEY = 'comments'
MENTIONS_KEY = 'mentions'
DESCRIPTION_KEY = 'description'
NUMBER_OF_COMMENTS_KEY = 'number_of_comments'
NUMBER_OF_RETWEETS_KEY = 'number_of_retweets'
NUMBER_OF_MENTIONS_KEY = 'number_of_mentions'
NUMBER_OF_FOLLOWING_KEY = 'number_of_following'
NUMBER_OF_LIKES_KEY = 'number_of_likes'
INTERACTION_VECTOR_KEY = 'interaction_vector'
SIMILARITY_VECTOR_KEY = 'similarity_vector'
DESCRIPTION_SIMILARITY = 'description_similarity'
FOLLOWING_SIMILARITY = 'following_similarity'
HASHTAGS_SIMILARITY = 'hashtags_similarity'
CATEGORIES_SIMILARITY = 'categories_similarity'

# interaction vector constants
COMMENTS_INTERACTION = 'comments_interaction'
RETWEETS_INTERACTION = 'retweets_interaction'
MENTIONS_INTERACTION = 'mentions_interaction'
FOLLOWING_INTERACTION = 'following_interaction'
LIKES_INTERACTION = 'likes_interaction'


# user info keys
LIST_OF_COMMENTS_KEY = 'list_of_comments'
LIST_OF_RETWEETS_KEY = 'list_of_retweets'
LIST_OF_MENTIONS_KEY = 'list_of_mentions'
LIST_OF_LIKES_KEY = 'list_of_likes'
LIST_OF_FOLLOWING_KEY = 'follow_to'
LIST_OF_HASHTAGS_KEY = 'hashtag_list'
LIST_OF_CATEGORIES_KEY = 'categories_list'


# API_KEYS
CONSUMER_ID_KEY = 'consumer_id_key'
CONSUMER_SECRET_KEY = 'consumer_secret_key'
ACCESS_TOKEN_KEY = 'access_token_key'
ACCESS_TOKEN_SECRET = 'access_token_secret'


# DATES
since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(60)
until = datetime.datetime.now(datetime.timezone.utc)


