#required modules
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import SyllableTokenizer
from nltk.corpus import stopwords
import re

#reading the given stopword list and converting them to csv
df1 = pd.read_csv('StopWords/StopWords_Auditor.txt',header=None, names=['words'])
df2 = pd.read_csv('StopWords/StopWords_Currencies.txt',encoding='latin-1',on_bad_lines='skip',header = None, names=['words'])
df3 = pd.read_csv('StopWords/StopWords_DatesandNumbers.txt',header=None,names=['words'])
df4 = pd.read_csv('StopWords/StopWords_GenericLong.txt',header=None, names=['words'])
df5 = pd.read_csv('StopWords/StopWords_Generic.txt',header=None, names=['words'])
df6 = pd.read_csv('StopWords/StopWords_Geographic.txt', header=None, names=['words'])
df7 = pd.read_csv('StopWords/StopWords_Names.txt',header=None, names=['words'])

#removing ' |' from dfs
df2['words'] = df2['words'].str.replace(r"\| [^\n][^']*" , '', regex=True)
df3['words'] = df3['words'].str.replace(r"\| [^\n][^']*" , '', regex=True)
df6['words'] = df6['words'].str.replace(r"\| [^\n][^']*" , '', regex=True)
df7['words'] = df7['words'].str.replace(r"\| [^\n][^']*" , '', regex=True)

#converting the stopwords to lowercase
df1['words'] = df1['words'].str.lower()
df2['words'] = df2['words'].str.title()
df3.loc[:88,'words'] = df3.loc[:88,'words'].str.lower()
df5['words'] = df5['words'].str.lower()
df6['words'] = df6['words'].str.title()
df7['words'] = df7['words'].str.title()

#converting the word column of dfs to list
auditors = df1['words'].values.tolist()
currencies = df2['words'].values.tolist()
datesNum = df3['words'].values.tolist()
genLong = df4['words'].values.tolist()
gen = df5['words'].values.tolist()
geographic = df6['words'].values.tolist()
names = df7['words'].values.tolist()

#the whole customized stop word list 
stop_words = auditors + currencies + datesNum + genLong + gen + geographic +names 

# reading the positive word and negative word text file converting them to list
positive_words = pd.read_csv('MasterDictionary/positive-words.txt',encoding='latin-1',on_bad_lines='skip',header = None, names=['words'])
positive_words = positive_words['words'].values.tolist()
negetive_words = pd.read_csv('MasterDictionary/negative-words.txt',encoding='latin-1',on_bad_lines='skip',header = None, names=['words'])
negetive_words = negetive_words['words'].values.tolist()

#required output 
output = pd.read_csv('Output Data Structure.csv')

#for customised stop words
def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    doc = tokenizer.tokenize(text)
    filtered_tokens = []
    for token in doc:
        if token not in stop_words:
            filtered_tokens.append(token)
    
    return filtered_tokens 

#for nltk stop words
stopWords = set(stopwords.words('english'))
def cleaning(text):
    tokenizer = RegexpTokenizer(r'\w+')
    doc = tokenizer.tokenize(text)
    filtered_tokens = []
    for token in doc:
        if token not in stopWords:
            filtered_tokens.append(token)
    return filtered_tokens 

#remove punctuation
def remove_punk(text):
    tokenizer = RegexpTokenizer(r'\w+')
    doc = tokenizer.tokenize(text)
    return doc

#find number of common words between two lists
def countCommon(list1, list2):
    res = list(set(list1).intersection(list2))
    return len(res)

#calculates the number of complex words
def complexWordCount(doc):
    count =0
    SSP = SyllableTokenizer()
    for token in doc:
        syllable = len(SSP.tokenize(token))
        if syllable > 2:
            count = count+1
    return count

#calculates number of syllable per word
def syllablePerWord(doc):
    total_syllables = 0
    SSP = SyllableTokenizer()
    for token in doc:
        total_syllables = total_syllables + len(SSP.tokenize(token))
    result = total_syllables/len(doc)
    return result


#calculate number of pronouns
def calculatePronoun(text):
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(text)
    return(len(pronouns))

#finds number of characters
def characterLen(doc):
    res = ''.join(doc)
    return len(res)


#calculating the varriables
for i in range(output.shape[0]):
    with open(f"data/{i+37}.txt") as f:
        content = f.read()
        
    sent_no = len(sent_tokenize(content))
    total_words = len(remove_punk(content))
    preprocessed_text = preprocess(content)
    total_words_after_clean = len(preprocessed_text)
    
    positive_score = countCommon(positive_words, preprocessed_text)
    negetive_score = countCommon(negetive_words, preprocessed_text)
    polarity_score = (positive_score-negetive_score)/((positive_score+negetive_score)+0.000001)
    subjectivity_score = (positive_score+negetive_score)/((total_words_after_clean)+0.000001)
    
    #
    
    avg_sent_len = total_words/sent_no
    
    complex_count = complexWordCount(remove_punk(content))
    
    percentage_cmplx_word = (complex_count*100)/total_words
    
    fog_index = 0.4*(avg_sent_len + percentage_cmplx_word)
    
    avg_no_word_per_sent = avg_sent_len
    
    word_count = len(cleaning(content))
    
    syllable_per_word = syllablePerWord(remove_punk(content))
    
    pronouns_count = calculatePronoun(content)
    
    avg_word_len = characterLen(remove_punk(content))/total_words
    
    
    output['POSITIVE SCORE'][i]=positive_score
    output['NEGATIVE SCORE'][i]=negetive_score
    output['POLARITY SCORE'][i]=polarity_score
    output['SUBJECTIVITY SCORE'][i]=subjectivity_score
    output['AVG SENTENCE LENGTH'][i]= avg_sent_len
    output['PERCENTAGE OF COMPLEX WORDS'][i]=percentage_cmplx_word
    output['FOG INDEX'][i]= fog_index
    output['AVG NUMBER OF WORDS PER SENTENCE'][i]= avg_no_word_per_sent
    output['COMPLEX WORD COUNT'][i]= complex_count
    output['WORD COUNT'][i]= word_count
    output['SYLLABLE PER WORD'][i]= syllable_per_word
    output['PERSONAL PRONOUNS'][i]= pronouns_count
    output['AVG WORD LENGTH'][i]= avg_word_len


#converting the output csv to excel
output.to_excel('Output Data Structure.xlsx')