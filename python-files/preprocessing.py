import nltk
from nltk.stem import WordNetLemmatizer
import re
import itertools
import html.parser


lemmatizer = WordNetLemmatizer()


re_http = r'https?.*?( |$)'
re_at = r'@.*?( |$)'
re_number = r'[0-9]+[.]?'
re_space = r'\s+'
re_remove_punctuation = r'[^a-zA-Z ]'
re_hash= r'#.*?( |$)'



apostrophe_lookup = {"'s": ' is', "'re": " are", "'d" :" would", "'ll":" will", "'ad": " had", "'t":" it", "'m": " am", "'ve":" have", "won't":"will not" ,"shan't":"shall not", "n't":" not"}





def lemmatizing(sent):
    l = []
    for i in sent.split():
        l.append(lemmatizer.lemmatize(i))
    return l


def basic_cleaning(string):
    sent = html.parser.unescape(string)
    sent = re.sub( re_http, "", sent )
    sent = re.sub( re_at, "", sent )
    sent = re.sub( re_number, "", sent )
    sent = re.sub( re_space, " ", sent )
    sent = re.sub(re_remove_punctuation, " ", sent)
    sent = ''.join( ''.join( s )[:2] for _, s in itertools.groupby(sent))
    for a in apostrophe_lookup.keys():
        if a in sent:
            sent= sent.replace(a, apostrophe_lookup[a])

    words = sent.split()
    new = []
    for x in words:
        if x.startswith('#'):
            x  = x.replace("#","")
            x_ = " ".join( re.findall( '[A-Z][^A-Z]*', x ) )
            if x_:
                x = x_
            x = x.split( " " )
            for ww in x:
                new.append(ww)
        else:
            new.append(x)
    sent = " ".join(new)
    return sent.lower().strip()


def preprocessing_stage1(string):
    sent=basic_cleaning(string)
    sent = lemmatizing(sent)
    return sent




def preprocessing_for_pos_tags(features):
    tweet_tags = []
    for t in features:
        tags_words = nltk.pos_tag( nltk.word_tokenize( t ) )
        tags = [x[1] for x in tags_words]
        tag_str = " ".join( tags )
        tweet_tags.append( tag_str )

    return tweet_tags
    
    
    
def preprocessing_stage2(string):
    sentence = re.sub( re_http, "URLHERE", string )
    sentence = re.sub( re_at, "MENTIONHERE", sentence )
    sentence = re.sub( re_hash, "HASHTAGHERE", sentence )
    sentence = re.sub( re_number, "", sentence )
    sentence = re.sub( re_space, " ", sentence )
    sentence = re.sub( re_remove_punctuation, " ", sentence )
    return sentence.lower()   
    






















