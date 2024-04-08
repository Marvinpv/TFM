from flask import Flask

app = Flask('APP')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"