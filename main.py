from flask import Flask, request, render_template, redirect, session
from model.model import predict

app = Flask(__name__, template_folder="website/templates", static_folder='website/public', static_url_path="")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=["POST"])
def predict2():
    result="computer generated"
    if predict(request.form.get('text')) == "OR":
        result="human-made"
    else:
        result="computer generated"
    return render_template("result.html", result=result, text=request.form.get('text'))

app.run(port=8000) #this will host this code on http://localhost:8000