#%%
import pandas as pd
import re 
import sklearn as sk
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


#%%
#read the data
job_description = pd.read_csv(r'..\data\Job description dataset\monster_com-job_sample.csv')
#print(job_description.head())
print(job_description.columns)
print(len(job_description))

resume = pd.read_csv(r'../data/Resume dataset/Resume/Resume.csv')
print('resume dataset loaded...')
# Look at only the ID, Category, and the first 100 characters of the resume
print(resume[['ID', 'Category', 'Resume_str']].head())

# Or even better, just see how many resumes you have
print("Total Resumes to clean: ", len(resume))


#%%
def clean_text(original_text):
    #covert all letter into lower case
    lower_text = original_text.lower()
    #removing all the special character or than space[ ]
    remove_special_char = re.sub(r'[^a-z\s]',' ',lower_text)

    words = remove_special_char.split() # spliting all the words so that we can filter out all the unecessary words
    filtered_words = [w for w in words if w not in stopwords]

    lemmztized_words = [lemmatizer.lemmatize(i, pos='v') for i in filtered_words ]
    
    
    return " ".join(lemmztized_words)

sample = "I AM mahEk JunnedI. *ANd Im %Working ON() A ;PRoject!!!, and my phn number is : 243567. i was recently having trouble opening my code's too."
#sample = "The players are running and jumping over hurdles, devlopment, managing, programming, testing"
result = clean_text(sample)
print(result)

#%%
resume.drop('Resume_html', axis =1, inplace= True)

#%%
print('The resume dataset is currently being cleaned.....this might take some time:)')
resume['Cleaned_Resume_str'] = resume['Resume_str'].apply(clean_text)
print('The resume_str has been cleaned, Thank you for your paitence.')
print(resume[['Resume_str', 'Cleaned_Resume_str']].head())


#%%
resume.to_csv('../data/Cleaned_Resumes.csv', index=False)


#%%
job_description.drop(['country', 'country_code', 'date_added' , 'has_expired'], axis = 1,inplace =True)

#%%
job_description.drop(['job_board','organization','page_url'], axis= 1, inplace = True)

# %%

print("Job-Description dataset is being cleaned...")
job_description['Cleaned_job_description'] = job_description['job_description'].apply(clean_text)
print("Job-Description is cleaned :))")
print(job_description[['job_description','Cleaned_job_description']].head())

# %%
job_description.to_csv('../data/Cleaned_job_description_dataset.csv')


# %%
print(job_description.columns)
# %%
