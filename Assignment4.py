# Import necessary libraries
import pyodbc
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import os

# Step 1: Connect to Azure SQL Database and Retrieve Data
def get_data_from_sql(server, database, username, password, table_name):
    driver = '{ODBC Driver 17 for SQL Server}'
    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    query = f'SELECT * FROM {table_name}'
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Step 2: Authenticate and Set Up Azure Cognitive Services Client
def authenticate_client(endpoint, key):
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

# Step 3: Perform Sentiment Analysis
def batch_documents(documents, batch_size=10):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def sentiment_analysis_example(client, documents):
    sentiments = []
    sentiment_scores = []
    for batch in batch_documents(documents):
        response = client.analyze_sentiment(documents=batch)
        for doc in response:
            if not doc.is_error:
                sentiments.append(doc.sentiment)
                sentiment_scores.append(doc.confidence_scores.positive - doc.confidence_scores.negative)
    return sentiments, sentiment_scores


# Step 4: Visualize Sentiment Analysis Results
def visualize_sentiments(data):
    # Bar Chart
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Sentiment', data=data)
    plt.title('Sentiment Distribution')

    # Donut Chart
    plt.subplot(1, 2, 2)
    sentiment_counts = data['Sentiment'].value_counts()
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Sentiment Distribution (Donut Chart)')
    plt.tight_layout()
    plt.show()

# Step 5: Save Results Back to Azure SQL Database
def save_to_sql(data, connection_string, table_name):
    try:
        engine = create_engine(connection_string)
        data.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data successfully saved to table {table_name} in the database.")
    except Exception as e:
        print(f"Error saving data to SQL: {e}")

# Step 6: Save Results to Local Machine
def save_to_local(data, file_name):
    try:
        data.to_csv(file_name, index=False)
        print(f"Data successfully saved to local file {file_name}.")
    except Exception as e:
        print(f"Error saving data to local file: {e}")

# Main function to execute all steps
def main():
    ## Azure SQL Database details
    server = 'server-akshai.database.windows.net'
    database = 'Infrastructure_AI '
    username = 'admin003'
    password = 'Unnikuttan@2017'
    table_name = 'final_yelp_labelled_sentiments'

    # Azure Cognitive Services details
    endpoint = 'https://assignmentd.cognitiveservices.azure.com/'
    key = '7a0684ec54aa495590d5e1562f98c969'

    # Retrieve data from SQL Database
    data = get_data_from_sql(server, database, username, password, table_name)

    # Authenticate Azure Cognitive Services client
    client = authenticate_client(endpoint, key)

    # Perform sentiment analysis
    documents = data['Cleaned_Text'].tolist()
    sentiments, sentiment_scores = sentiment_analysis_example(client, documents)
    data['Sentiment'] = sentiments
    data['SentimentScore'] = sentiment_scores

    # Visualize the results
    visualize_sentiments(data)

    # Display the DataFrame with Sentiment Analysis Results
    print(data)

    # Save the results back to SQL Database
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server'
    save_to_sql(data, connection_string, 'yelp_labelled_sentiments')


# Run the main function
if __name__ == "__main__":
    main()
