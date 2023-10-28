# imports and initializations
from flask import Flask, request, jsonify

# for the translation part
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import openai
import time
from translate import Translator

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the API. You can use the API via http://localhost:5001/fetch, POST request of JSON type {"user_input": "something"}', 200


# Specify Dhivehi as the source language (dv)
translator = Translator(from_lang="dv", to_lang="en")


# openai api key
openai.api_key = ''  # your key here

# for fetching the documents from the search engine
API_KEY: str = ''
SEARCH_ENGINE_ID: str = ''
URL: str = 'https://www.googleapis.com/customsearch/v1'

# for date time
# Get the current date and time
current_date_time = datetime.datetime.now()
# Get the current date
current_date = current_date_time.date()


@app.route('/fetch', methods=['POST'])
def search():
    try:
        user_input = request.json.get('user_input')

        translation = translator.translate(user_input)

        converted_for_prompting = ''

        the_prompt = f"""
        can you rephrase the following to a question general question? {translation}.
        """
        # Send the prompt to LLM API to generate the response
        try:
            response = openai.Completion.create(
                model='gpt-3.5-turbo-instruct',
                prompt=the_prompt,
                temperature=0.0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[' Human:', ' AI:']
            )
            choices = response.get('choices')[0]
            bot_response = choices.get('text')
            converted_for_prompting = bot_response
        except Exception as e:
            print('ERROR: ', e)

        print(f'User: {user_input}')
        print(f'For prompting: {converted_for_prompting}')

        # Define the prompt and user input
        the_prompt = f"""
            Role: You are a factual and helpful assistant to aid users in the lateral reading task. You will receive a segment of text (Text:), 
            and you need to raise five important, insightful, diverse, simple, factoid questions that may arise to a user when reading the text but are not answered by the text 
            (Question1:, Question2:, Question3:, Question4:, Question5:). The questions should be suitable as meaningful queries to a search engine like Bing.
            Include a question regarding current status. Like the following. Question: What is the current situation or status in this context?
            Your questions will motivate users to search for relevant documents to better determine whether the given text contains misinformation. 
            Additionally, please ask a question related to the current status or situation with reference to the text.
            User: Text: {converted_for_prompting}
            Carefully choose insightful and atomic lateral reading questions not answered by the above text, ensuring that the queries are self-sufficient 
            (Do not have pronouns or attributes relying on the text; they should be fully resolved and make complete sense independently).
        """
        # Send the prompt to LLM API to generate the response
        try:
            response = openai.Completion.create(
                model='gpt-3.5-turbo-instruct',
                prompt=the_prompt,
                temperature=0.0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[' Human:', ' AI:']
            )

            choices = response.get('choices')[0]
            bot_response = choices.get('text')

            # Extract questions using regular expression
            lateral_questions = re.findall(
                r'Question\d+: (.*?)\n', bot_response)

        except Exception as e:
            print('ERROR: ', e)

        lateral_questions.append(converted_for_prompting)

        # our data structure
        lateral_structure = []
        for i in range(0, len(lateral_questions)-1):
            item = {
                'lateral_question': f'{lateral_questions[i]}',
                'links': [],
                'text': '',
                'lateral_answer': ''
            }
            lateral_structure.append(item)

        for data in lateral_structure:
            search_query = data['lateral_question']
            params: dict = {
                'q': search_query,
                'key': API_KEY,
                'cx': SEARCH_ENGINE_ID
            }
            try:
                response = requests.get(URL, params=params)
            except:
                pass
            results = response.json()
            link_set = []
            # we will only store 8 link for each lateral question for now  # increasing this will improve accuracy
            for i in range(0, 8):
                # but we will be using 3 that works, because some links tend to be broken
                if 'items' in results:
                    link_set.append(results['items'][i]['link'])
            link_set = list(set(link_set))  # to make it distinct
            data['links'] = link_set

        # can cythonize this part

        total_links = []  # List to store the number of links for each lateral_structure entry

        for entry in lateral_structure:
            total_links.append(len(entry['links']))

        total_links = []  # List to store the number of links for each lateral_structure entry

        for entry in lateral_structure:
            total_links.append(len(entry['links']))

        for i, entry in enumerate(lateral_structure):
            segment_set = []  # List to store processed text for each entry in the lateral structure
            best_similarity = -1  # Variable to store the best cosine similarity found
            best_text = None  # Variable to store the text with the best cosine similarity

            processed_links = 0  # Counter for processed links
            # Loop through all links for each entry
            for j in range(total_links[i]):
                if processed_links >= 3:
                    break  # Stop processing links if 3 have already been processed
                try:
                    response = requests.get(entry['links'][j])

                    if response.status_code == 200:
                        html_content = response.text
                        soup = BeautifulSoup(html_content, 'html.parser')
                        all_text = soup.get_text()

                        # Truncate to 250 tokens for now; you can modify this logic based on your requirements
                        words = all_text.split()

                        if len(words) > 250:
                            words = words[:250]

                        text_string_250_tokens = ' '.join(words)

                        # Find the last comma (or dot) in the trimmed text
                        last_comma_index = max(text_string_250_tokens.rfind(
                            ','), text_string_250_tokens.rfind('.'))

                        # Remove characters after the last comma (or dot)
                        if last_comma_index != -1:
                            final_text = text_string_250_tokens[:last_comma_index + 1]
                        else:
                            # No comma (or dot) found
                            final_text = text_string_250_tokens

                        segment_set.append(final_text)
                        processed_links += 1  # Increment the processed links counter

                        # Calculate cosine similarity between lateral_question and processed text
                        vectorizer = TfidfVectorizer()
                        vectors = vectorizer.fit_transform(
                            [entry['lateral_question'], final_text])
                        similarity = cosine_similarity(
                            vectors[0], vectors[1])[0][0]

                        # Update best_text if current text has higher cosine similarity
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_text = final_text

                    else:
                        print(
                            f"Failed to retrieve the web page {entry['links'][j]}. Status code:", response.status_code)

                except Exception as e:
                    print(f"An error occurred: {e}")

            # Store the text with the best cosine similarity
            if best_text is not None:
                lateral_structure[i]['text'] = best_text
            else:
                lateral_structure[i]['text'] = None  # No suitable text found\

        # Define the prompt and user input

        for data in lateral_structure:
            the_prompt = f"""
            Role: You are a factual and helpful assistant to aid users in the lateral reading task. You will receive a segment
            of text (Text:), and I would like you to carefully read it and see if you can answer that question using an assertive tone.
            I will be providing a link, in your verdict mention that you deduced the answer based on that reference link. Do not start with your answer with a 'yes' or 'no'.

            User:
            question: {data['lateral_question']}
            text: {data['text']}
            reference link: {data['links']}

            AI: 
            """
            # Send the prompt to LLM API to generate the response
            try:
                response = openai.Completion.create(
                    model='gpt-3.5-turbo-instruct',
                    prompt=the_prompt,
                    temperature=0.0,
                    max_tokens=250,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0.6,
                    stop=[' Human:', ' AI:']
                )
                choices = response.get('choices')[0]
                bot_response = choices.get('text')

                data['lateral_answer'] = bot_response

            except Exception as e:
                print('ERROR: ', e)

            time.sleep(21)

        main_prompt_lateral_context = []

        for data in lateral_structure:
            segment = f"""
            A question from lateral reading that was raised was {data['lateral_question']}.
            And the answer by a reader was {data['lateral_answer']}.
            The reference link that lead to the answer was {data['links']}
            """
            main_prompt_lateral_context.append(segment)

        main_prompt_lateral_context_string = '\n\n'.join(
            main_prompt_lateral_context)

        the_prompt = f"""
        Role: You are a factual and helpful assistant designed to read and cohesively summarize segments from different
        relevant document sources to answer the question at hand. Your answer should be informative but no more than hundred
        words. Your answer should be concise, easy to understand and should only use information from the provided
        relevant segments but combine the search results into a coherent answer. Do not repeat text and do not include
        irrelevant text in your answers. Use an unbiased and journalistic tone. Make sure the output is in plaintext.
        Attribute each sentence with proper citations using the document number with the [doc_number] notation
        (Example: "Hydroxychloroquine is not a cure for COVID-19 [1][3]."). Ensure each sentence in the answer is
        properly attributed. Ensure each of the documents is cited at least once. If different results refer to different
        entities with the same name, cite them separately.
        User: My question is {converted_for_prompting}. Cohesively and factually summarize the following documents to answer my
        question. 

        {main_prompt_lateral_context_string}

        Also note current date is {current_date}. And please provide the links as reference at the end. 
        And try to determine the time of the documents and its relevance to current date {current_date}.
        You can use the grammar or the tense to determine whether its a past event to the current date or not. This is absolutely crucial.
        And remember your answer should be informative but no more than hundred words
        """
        # Send the prompt to LLM API to generate the response
        try:
            response = openai.Completion.create(
                model='gpt-3.5-turbo-instruct',
                prompt=the_prompt,
                temperature=0.0,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[' Human:', ' AI:']
            )
            choices = response.get('choices')[0]
            bot_response = choices.get('text')
            print(f'User: {user_input}')
            print(f'LLM Response: {bot_response}')
            response_data = {'Response': bot_response}
            return jsonify({'result': response_data}), 200
        except Exception as e:
            print('ERROR: ', e)

    except:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
