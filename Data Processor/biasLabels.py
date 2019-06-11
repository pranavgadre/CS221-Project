import pandas as pd
import numpy as np
import os
import glob
'''
os.chdir('all_the_news')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
all_articles = pd.concat([pd.read_csv(f) for f in all_filenames ])

newData = all_articles[['publication', 'content']]
publishers = all_articles[['publication']]
'''
# DATASET 1

# Reading and fetching desired columns
data1 = pd.read_csv('all-the-news/articles1.csv')
newData1 = data1[['content']]

# Creating new columns for political leaning
li1 = []
for i in range(0, len(data1)):
    source = data1.iloc[i]['publication']
    if source == 'Atlantic':
        li1.append(-1)
    elif source == 'CNN':
        li1.append(-2)
    elif source == 'Business Insider':
        li1.append(-1)
    elif source == 'Breitbart':
        li1.append(2)
    elif source == 'New York Times':
        li1.append(-1)
newData1['leaning'] = li1

# DATASET 2

data2 = pd.read_csv('all-the-news/articles2.csv')
newData2 = data2[['content']]

li2 = []
for i in range(0, len(data2)):
    source = data2.iloc[i]['publication']
    if source == 'Atlantic':
        li2.append(-1)
    elif source == 'Fox News':
        li2.append(2)
    elif source == 'Talking Points Memo':
        li2.append(-2)
    elif source == 'Buzzfeed News':
        li2.append(-1)
    elif source == 'New York Post':
        li2.append(1)
    elif source == 'Guardian':
        li2.append(-1)
    elif source == 'National Review':
        li2.append(2)
    
newData2['leaning'] = li2

# DATASET 3

data3 = pd.read_csv('all-the-news/articles3.csv')
newData3 = data3[['content']]

li3 = []
for i in range(0, len(data3)):
    source = data3.iloc[i]['publication']
    if source == 'Guardian':
        li3.append(-1)
    elif source == 'Washington Post':
        li3.append(-1)
    elif source == 'NPR':
        li3.append(-1)
    elif source == 'Reuters':
        li3.append(0)
    elif source == 'Vox':
        li3.append(-2)
newData3['leaning'] = li3

# Creating the new csv's (publisher, content, leaning)

newData1.to_csv('articles_with_biases1.csv')
newData2.to_csv('articles_with_biases2.csv')
newData3.to_csv('articles_with_biases3.csv')
