import pandas as pd
import re 
import sklearn as sk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

#read the data
job_description = pd.read_csv('data/Job description dataset/monster_com-job_sample.csv')
job_description.drop(['country', 'country_code', 'date_added' , 'has_expired','job_board','organization','page_url'], axis = 1,inplace =True)
resume = pd.read_csv('data/Resume dataset/Resume/Resume.csv')

#Data cleaning and preprocessing
def clean_text(original_text):
    #covert all letter into lower case
    lower_text = original_text.lower()
    #removing all the special character or than space[ ]
    remove_special_char = re.sub(r'[^a-z\s]',' ',lower_text)

    words = remove_special_char.split() # spliting all the words so that we can filter out all the unecessary words
    filtered_words = [w for w in words if w not in stopwords]

    lemmztized_words = [lemmatizer.lemmatize(i, pos='v') for i in filtered_words ] 
    
    
    return " ".join(lemmztized_words)

#clean job titles
def clean_job_titles(title):
    if pd.isna(title):
        return ""  # Return an empty string for NaN values
    
    title_str = str(title).lower()  # Convert to lowercase for uniformity

    #if 'jobid:' in title_str:
       # title_str = title_str.split('jobid:')[-1].strip()  # Keep only the part after 'JobID:'

    for tigger in ["job in", "job application","-","|"]:
        if tigger in title_str:
            title_str = title_str.split(tigger)[0].strip()  # Keep only the part before the trigger

    if 'body{' in title_str:
        title_str = title_str.split('body{')[0].strip()  # Remove any unnecassary HTML tags or styles

    return title_str


print('The resume dataset is currently being cleaned.....this might take some time:)')
resume['Cleaned_Resume_str'] = resume['Resume_str'].apply(clean_text)
print('The resume_str has been cleaned, Thank you for your paitence.')


resume.to_csv('data/Cleaned_Resumes.csv', index=False)

print("Job-Description dataset is being cleaned...")
job_description['Cleaned_job_description'] = job_description['job_description'].apply(clean_text)
job_description['Cleaned_job_title'] = job_description['job_title'].apply(clean_job_titles)
print("Job-Description is cleaned :))")

job_description.to_csv('data/Cleaned_job_description_dataset.csv', index=False)

