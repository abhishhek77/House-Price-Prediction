from flask import Flask, render_template, request
from src.predict import predict_price

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    price = None

    if request.method == "POST":
        house = {
            "area_sqft": float(request.form["area_sqft"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "age_years": int(request.form["age_years"]),
            "garage_spaces": int(request.form["garage_spaces"]),
            "location": request.form["location"],
        }

        price = predict_price(house)

    return render_template("index.html", price=price)


if __name__ == "__main__":
    app.run(debug=True)