import jsonlines
import re

dataset_name = "sample-1M.jsonl"

filtered_dataset = "filtered_dataset_full.jsonl"
temp = ""
f = open(filtered_dataset, "w")

i = 0
# 734488 articles
"""
Operations: check number of aphex and decide to inlude them or not
Operations: you need an online service to view this article in its entirety <-- delete or not?


"""

"""
with jsonlines.open(dataset_name) as r:
    n_aphex=0
    n_aphex2=0
    for obj in r:
        if obj['media-type']=='News':
            article=''.join(obj['content'].splitlines()) #remove \n
            n_aphex+=article.count("'")
            n_aphex2+=article.count('"')
    print(n_aphex)
    print(n_aphex2)

"""
"""
Articles  673512
Articles with subscription  3293
Articles removed  286
Articles used  673226

"""



articles_with_subscription = 0
articles_removed = 0
with jsonlines.open(dataset_name) as r:
    art = 0
    for obj in r:
        i += 1
        if i % 10000 == 0:
            print(i)
            print(art)
            print("---------------")

        if obj['media-type'] == 'News':
            article = ' '.join(obj['content'].splitlines())  # remove \n
            article = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', article)  # remove links
            article=re.sub(r'[^@]+@[^@]+\.[^@]+',' ',article)#emails
            article=re.sub(r'@([A-Za-z0-9_]+)',' ',article) # Remove tweet usernames
            article = article.replace('\t',' ') # remove tabs
            article=article.replace('*',' ')
            article=article.replace('…',' ') # remove this character omg 
            article = article.replace('[', ' [ ')
            article = article.replace(']', ' ] ')
            article=article.replace('©','')

            article = re.sub('([.,?!:;"”“()&$])', r' \1 ', article)
            article = re.sub("(['])", r' \1 ', article)
            article = " ".join(article.split())  # remove duplicated spaces

            title=''.join(obj['title'].splitlines())
            title=re.sub('([.,?!:;"”“()&$])', r' \1 ', title)
            data = {"content": article.lower(), "title": title.lower()}
            final_string = ''

            #minimo 35 token

            if "subscription required an online service is needed to view this article" in data['content']:
                final_string = data['content'][
                               :data['content'].find('subscription required an online service is needed '
                                                     'to view this article')]
                articles_with_subscription += 1
                if final_string == '':
                    articles_removed += 1
                    continue
                if not final_string.endswith('.'):
                    final_string += '.'
            elif "sorry the page you requested either" in data['content']:
                continue
            elif "minutes ago julio pratama tranquillo barnetta s goal gives philadephia early lead" in data['content']:
                continue
            elif "sorry the page you requested either doesn" in data['content']:
                continue
            else:
                if data['content']=='' or data['content']==' ':
                    continue
                final_string = data['content']
            if not data['title'].endswith('.'):
                data['title'] = data['title'] + '.'

            final_string += '\t'
            final_string += data['title']
            final_string += '\n'

            f.write(final_string)
            art += 1

f.close()
print("Articles ",art)
print("Articles with subscription ",articles_with_subscription)
print("Articles removed ", articles_removed)
print("Articles used ",art-articles_removed)
