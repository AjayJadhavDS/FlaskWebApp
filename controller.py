from flask import Flask, render_template, request
from wtforms import Form, FloatField, validators
#from compute import compute
import pandas as pd
from sklearn.externals import joblib


app = Flask(__name__)

# Model
class InputForm(Form):
	r = FloatField(validators=[validators.InputRequired()])
	
# View
@app.route('/hw1', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
		years_of_experience = form.r.data
		lin_reg = joblib.load("linear_regression_model.pkl")
		s = lin_reg.predict([[years_of_experience]]).tolist()
		#s = compute(df['r_'].values,df['p_'].values)
		return render_template("view_output.html", form=form, s=s)
    else:
		return render_template("view_input.html", form=form)

if __name__ == '__main__':
	
	print ('Model loaded')
	app.run(debug=True)