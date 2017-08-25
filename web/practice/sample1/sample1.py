#coding:utf-8
import flask
import wtforms
from flask import render_template
from flask_socketio import SocketIO
from flask.ext.wtf import Form
from flask import request

class ChildrenForm(Form):

    input_field=wtforms.StringField('input field')

app = flask.Flask(__name__)
# socketio=SocketIO(app)
app.config['WTF_CSRF_ENABLED'] = False

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method=='GET':
        childrenForm=ChildrenForm()
        return render_template('home.html', form=childrenForm)

    elif request.method=='POST':
        childrenForm=ChildrenForm()
        print childrenForm.data
        return ''

if __name__ == '__main__':
    # socketio.run(app, '0.0.0.0', 8000)
    app.run()