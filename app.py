from flask import Flask, request, jsonify, render_template
import pickle
import neattext.functions as nfx

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.sav", "rb"))
countv = pickle.load(open('countvect.pkl','rb'))

def transform_text(text):
    text=nfx.remove_special_characters(text)
    text=nfx.remove_puncts(text)
    text=nfx.clean_text(text)
    
    return text


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    query1=request.form["title"]
    query2=request.form["author"]
    query3=request.form["total"]
    data=query1+' '+query2+' '+query3
 
    transformed_data = transform_text(data)
    vector_input = countv.transform([transformed_data])
    result = model.predict(vector_input)

    if result:
        prediction="This news is fake!"
    else:
        prediction="This news is real!"

    
    return render_template("index.html",output=prediction,query1=request.form["title"],query2=request.form["author"],query3=request.form["total"])

if __name__ == "__main__":
    flask_app.run(debug=True)