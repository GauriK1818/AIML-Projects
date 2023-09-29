import re
import pandas as pd

def preprocess(data):
    pattern = '\d{1,2}\/\d{1,2}\/\d{2},\s\d{1,2}:\d{2}\s(?:AM|PM)'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message':messages, 'message_date': dates})


    df['message_date'] = df['message_date'].apply(lambda x: re.sub(r'[\u200B-\u200D\uFEFF]', ' ', x))

# Update the format string to include 'AM/PM'
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?:\s)', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        
        else:
            users.append('group_notification')
            messages.append(entry[0])
        
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['year']=df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day']=df['date'].dt.day
    df['hour']=df['date'].dt.hour
    df['minute']=df['date'].dt.minute
    return df
