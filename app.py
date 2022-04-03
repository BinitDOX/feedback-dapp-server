from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from db import db_init, db
from db_models import Img
from models.classification import get_class
from models.rating import get_rating

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


@app.route('/')
@cross_origin()
def hello_world():
    return 'Hello, World!'


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    print(request.form['address'])
    pic = request.files['image']
    address = request.form['address']
    if not pic:
        return 'No picture uploaded!', 400

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400

    img = Img(img=pic.read(), address=address, name=filename, mimetype=mimetype, )
    db.session.add(img)
    db.session.commit()

    return 'Img Uploaded!', 200


@app.route('/image-by-address/<string:address>')
@cross_origin()
def get_img_by_id(address):
    img = Img.query.filter_by(address=address).first()
    if not img:
        return 'Image Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)
    
    
@app.route('/image-by-id/<int:id>')
@cross_origin()
def get_img_by_address(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Image Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)
    
    
@app.route('/feedback-classify', methods=['POST'])
@cross_origin()
def classify():
    print(request.form['content'])
    c = get_class(request.form['content'])
    return c, 200
    
    
@app.route('/feedback-rating', methods=['POST'])
@cross_origin()
def rating():
    req = request.form['content']
    print(req)
    if type(req) is list:
        r = get_rating(request.form['content'])
        print(r)
        return r, 200
    else:
        return 'Request expected to be a list of feedbacks', 400
    
    
if __name__ == '__main__':
    #app.run(host="127.0.0.1", port=5001, debug=True)
    #app.run()
    app.run(host="192.168.0.139", port=5001, debug=False)
