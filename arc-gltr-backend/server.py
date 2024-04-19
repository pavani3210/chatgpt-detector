import argparse
import datetime
import os
from flask import Flask, request, session
from flask_cors import CORS
import logging
from backend import AVAILABLE_MODELS
import json
import datetime
from backend import api_gptzero

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = '/'
logger = logging.getLogger('')
ALLOWED_EXTENSIONS = set(['txt', 'docx', 'pdf', 'doc'])
CONFIG_FILE_NAME = 'lmf.yml'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

projects={}
def application(environ, start_response):
  if environ['REQUEST_METHOD'] == 'OPTIONS':
    start_response(
      '200 OK',
      [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'POST'),
      ]
    )
    return ''
class Project:
    def __init__(self, LM, config):
        self.config = config
        self.lm = LM()

def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].configs
    return res

@app.route('/hello', methods=['GET'])
def test():
    return "hello world"

@app.route('/stats', methods=['GET'])
def getStats():
    with open('upload-count.json', 'r') as f:
        data = json.load(f)
    return data

@app.after_request
def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

def count():
    data={}
    with open('upload-count.json', 'r') as f:
        data = json.load(f)
    today = datetime.date.today()
    p_day, p_week, p_month = today.strftime("%m-%d-%Y"), str(today.isocalendar()[1])+"-"+str(today.year), today.strftime("%m-%Y")
    data["day"][p_day] = 0 if p_day not in data["day"] else data["day"][p_day]+1
    data["week"][p_week] = 0 if p_week not in data["week"] else data["week"][p_week]+1
    data["month"][p_month] = 0 if p_month not in data["month"] else data["month"][p_month]+1
    with open('upload-count.json', 'w') as f:
        json.dump(data, f)
    print(data)
    
@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER)
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file'] 
    filename = file.filename  # Get the filename
    project = request.form['project']
    count()
    res = {}
    if project == 'gptzero':
        res = api_gptzero.extract_files(file)
    else: 
        if project in projects:
            p = projects[project] 
            res = p.lm.extract_files(p, file, topk=20)
    return res

parser = argparse.ArgumentParser()
parser.add_argument("--model1", default='BERT')
parser.add_argument('--model2', default='gpt-2')
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="0.0.0.0")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))

parser.add_argument("--no_cors", action='store_true')

args, _ = parser.parse_known_args()

try:
    model1 = AVAILABLE_MODELS[args.model1]
except KeyError:
    print(f"Model {args.model1} not found. Make sure to register it.")
    print("Loading BERT instead.")
    model1 = AVAILABLE_MODELS['BERT']
    
try:
    model2 = AVAILABLE_MODELS[args.model2]
except KeyError:
    print(f"Model {args.model2} not found. Make sure to register it.")
    print("Loading gpt-2 instead.")
    model2 = AVAILABLE_MODELS['gpt-2']

projects = {'BERT' : Project(model1, args.model1), 'gpt-2' : Project(model2, args.model2)}


if __name__ == '__main__':
    args = parser.parse_args()
    app.run(port=int(args.port), debug=not args.nodebug, host=args.address)
    
CORS(app, expose_headers='Authorization')