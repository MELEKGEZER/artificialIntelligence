from flask import Flask, render_template

app = Flask(__name__)

# Ana sayfa route'u
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)