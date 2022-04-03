from db import db


class Img(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.Text, unique=True, nullable=False)
    img = db.Column(db.Text, unique=True, nullable=False)
    name = db.Column(db.Text, nullable=False)
    mimetype = db.Column(db.Text, nullable=False)