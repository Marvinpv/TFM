from flask import Flask,render_template,jsonify

app = Flask('APP',static_url_path='/static')

@app.route("/")
def main_menu():
    return render_template('main_menu.html')

@app.route("/dataset")
def dataset_view():
    return render_template('dataset_view.html')

@app.route('/opciones')
def obtener_opciones():
    opciones = generar_opciones()
    return jsonify(opciones)

