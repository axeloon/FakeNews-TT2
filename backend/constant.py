USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"

#Start X Method
X_BASE_URL = "https://x.com"
X_LOGIN_BUTTON_XPATH = "//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/div[1]/div/div[3]/div[4]/a"
X_EMAIL_INPUT_XPATH = "//*[@id='layers']/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[4]/label/div/div[2]/div/input"
X_NEXT_BUTTON_XPATH = '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/button[2]'
X_SELECTOR_MODAL_HEADER = '[data-testid="ocfEnterTextTextInput"]'
X_USERNAME_INPUT_XPATH = '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div[2]/label/div/div[2]/div/input'
X_USERNAME_NEXT_BUTTON_XPATH = '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div/div/button'
X_PASSWORD_INPUT_XPATH = '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input'
X_LOGIN_SUBMIT_BUTTON_XPATH = '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/button'

#Fetch User Profile Method
X_PROFILE_LOADING_SELECTOR = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div'
X_PROFILE_CONTENT_XPATH = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div/div'
X_PROFILE_TWEETS_XPATH = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div'

#Process User Profile Method
HTML_PARSER = 'html.parser'
X_ARIA_LABEL_SELECTOR = 'aria-label'
X_VERIFIED_ICON_ARIA_LABEL = 'Verified account'
X_USER_NAME_DIV_CLASS = 'css-146c3p1 r-bcqeeo r-1ttztb7 r-qvutc0 r-37j5jr r-adyw6z r-135wba7 r-1vr29t4 r-1awozwy r-6koalj r-1udh08x'
X_JOIN_DATE_TEXT = 'Joined'
X_FOLLOWERS_SPAN_CLASS = 'css-1jxf684'
X_PROFILE_LABELS = ['Following', 'Followers', 'Subscriptions', 'Followed By']

# Extract Tweets Method
X_DATA_TESTID = 'data-testid'
X_TWEET_TEXT = 'tweetText'
X_USER_NAME = 'User-Name'
X_TWEET_IMAGE = 'tweetPhoto'

# Load More Tweets Method
def get_window_scroll_by(max_scroll_depth):
    return f'window.scrollBy(0, window.innerHeight / {max_scroll_depth})'

WINDOW_SCROLL_TO = 'window.scrollTo(0, document.body.scrollHeight)'

# Constantes para TweetService
X_TWEET_TEXT = 'tweetText'
X_USER_NAME = 'User-Name'
X_TWEET_IMAGE = 'tweetPhoto'
X_TWEET_IMAGE_PATH = 'tweets'

# Selectors
SPAN = 'span'
DIV = 'div'
TIME = 'time'
IMG = 'img'

# Emol
NEWSAPI_NOTICIA = 'https://newsapi.ecn.cl/NewsApi/emol/noticia/'
TABLE = "<table"
TABLA_EN_NOTICIA = "class=\"tablaennoticia\""
