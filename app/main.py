# import requirements needed
import os

from flask import Flask, request, redirect, url_for, render_template, session
from utils import get_base_url
from aitextgen import aitextgen
from happytransformer import HappyTextToText, TTSettings
from transformers import pipeline

from styleformer import Styleformer
import torch
import warnings

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12360
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url + 'static')

# ai_customersupport = aitextgen(model_folder="model/customer_support", to_gpu=False)
# ai_conversation = aitextgen(model_folder="model/human_conversation", to_gpu=False)
# ai_academic = aitextgen(model_folder="model/academic", to_gpu=False)

app.secret_key = os.urandom(64)

write_genre = ''


def genre_text_generation(genre):

    if genre == 'customer_support':
        file_dest = 'model/customer_support'

    if genre == 'daily_conversation':
        file_dest = 'model/human_conversation'

    if genre == 'academic':
        file_dest = 'model/academic'

    return file_dest


# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')


@app.route(f'{base_url}/generating')
def shakespeare():
    return render_template('generating.html')


@app.route(f'{base_url}/poe')
def poe():
    return render_template('Poe.html')


@app.route(f'{base_url}/team')
def team():
    return render_template('team.html')


@app.route(f'{base_url}/results/')
def results():
    if 'data' in session:
        data = session['data']
        return render_template('generating.html', generated=data)
    else:
        return render_template('generating.html', generated=None)


@app.route(f'{base_url}/grammar_fix/', methods=["POST"])
def grammar_fix():
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    #     beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)
    beam_settings = TTSettings(num_beams=5, min_length=1)

    #     input_text_1 = input("Enter: ")

    data = session['data']
    #     data = 'grammar: This sentences has has bads grammar.'
    print(data)
    output_data = happy_tt.generate_text(data, args=beam_settings)
    print(output_data.text)
    session['data'] = output_data.text
    print(session['data'])
    return redirect(url_for('results'))


@app.route(f'{base_url}/summarization/', methods=["POST"])
def summarization():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    #get the unprocessed data from session, name it with data
    data = session['data']
    
    #processing the original data
    output_data = summarizer(data,
                             max_length=130,
                             min_length=30,
                             do_sample=False)
    string_output = ''
    for i in output_data:
        string_output += i['summary_text']
    print(output_data)
    
    #Putting back the processed data to the session
    print(type(output_data))
    print(output_data)
    session['data'] = string_output
    
    #returning to the result page for update
    return redirect(url_for('results'))


@app.route(f'{base_url}/styletransfer/', methods=["POST"])
def styletransfer():
#     warnings.filterwarnings("ignore")


    #get the data from the web
#     data = 'The movie was really long and boring'
    data = session['data']
    
    
    from styleformer import Styleformer
    import torch
    import warnings
    warnings.filterwarnings("ignore")


    #uncomment for re-producability
    def set_seed(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(1234)


    
    sf = Styleformer(style = 0) 

    source_sentences = data
    output = sf.transfer(source_sentences)
    session['data'] = output

    #returning to the result page for update
    return redirect(url_for('results'))



@app.route(f'{base_url}/generate_text/', methods=["POST"])
def generate_text():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    print(prompt)
    genre_type = request.form['genre']
    print('genre type is: ', genre_type)
    story_genre = genre_text_generation(genre_type)
    print('file destination is: ', story_genre)
    ai = aitextgen(model_folder=story_genre, to_gpu=False)
    if prompt is not None:
        if genre_type == 'academic':
            generated = ai.generate(n=1,
                batch_size=1,
                prompt=str(prompt),
                max_length=240,
                temperature=0.7,
                top_p=0.9,
                return_as_list=True)
        if genre_type == 'customer_support':
            generated = ai.generate(n=1,
                batch_size=1,
                prompt=str(prompt),
                max_length=20,
                temperature=0.7,
                top_p=0.9,
                return_as_list=True)
        if genre_type == 'daily_conversation':
            generated = ai.generate(n=1,
                batch_size=1,
                prompt=str(prompt),
                max_length=40,
                temperature=0.7,
                top_p=0.9,
                return_as_list=True)

    data = {'generated_ls': generated}
    session['data'] = generated[0]
    return redirect(url_for('results'))


# Define results_poe() and generate_text_poe()
@app.route(f'{base_url}/results_poe/')
def results_poe():
    if 'data' in session:
        data = session['data']
        return render_template('Poe.html', generated=data)
    else:
        return render_template('Poe.html', generated=None)


@app.route(f'{base_url}/generate_text_poe/', methods=["POST"])
def generate_text_poe():
    prompt = request.form['prompt']
    if prompt is not None:
        generated = ai_poe.generate(n=1,
                                    batch_size=3,
                                    prompt=str(prompt),
                                    max_length=300,
                                    temperature=0.7,
                                    return_as_list=True)

    data = {'generated_ls': generated}
    session['data'] = generated[0]
    return redirect(url_for('results_poe'))


# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc9.ai-camp.dev/'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)
