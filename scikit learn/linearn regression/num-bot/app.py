
import decimal
from flask import Flask
from model.simpleLinearRegressing import model
from actions.sys_admin import task_list_files

# configure the decimal  precision
decimal.getcontext().prec = 2

app = Flask(__name__)

@app.route('/<int:name>/<cmd>')
def predict(name,cmd):
    response_data =''
    pred = decimal.Decimal(model.predict([[int(name)]])[0][0])
    if pred <= 2:
        try:
            response_data=response_data.join(task_list_files(cmd))
            response_data = response_data.replace('\n','<br>')
            
        except Exception as e:
            response_data = e
    else:
        response_data = pred
    return f"Bot response>> {response_data}"