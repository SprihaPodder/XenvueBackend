DB_URI = "mysql+mysqlconnector://admin:Spriha06032006@database-1.cpag2wi0e389.eu-north-1.rds.amazonaws.com:3306/database1"



# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

# app = Flask(__name__)

# DB_URI = "mysql+mysqlconnector://admin:Spriha06032006@database-1.cpag2wi0e389.eu-north-1.rds.amazonaws.com/database-1"

# # DB_URI = "mysql+mysqlconnector://admin:Spriha06032006@host:3306/database1"


# db = SQLAlchemy(app)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(80), unique=False, nullable=False)

#     def __repr__(self):
#         return '<User %r>' % self.username

# @app.route('/')
# def home():
#     return "connected to the db"

# if __name__ == '__main__':
#     app.run(debug=True)