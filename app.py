from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import io
import base64

from utils import predict_plant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicinal_plants.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# =========================
# CREATE UPLOAD FOLDER
# =========================

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# DATABASE
# =========================

db = SQLAlchemy(app)

# =========================
# LOGIN MANAGER
# =========================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = 'login'

login_manager.login_message_category = 'info'

# =========================
# USER MODEL
# =========================

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(80), unique=True, nullable=False)

    email = db.Column(db.String(120), unique=True, nullable=False)

    password_hash = db.Column(db.String(120), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):

        self.password_hash = generate_password_hash(password)

    def check_password(self, password):

        return check_password_hash(self.password_hash, password)

# =========================
# PREDICTION HISTORY MODEL
# =========================

class PredictionHistory(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )

    filename = db.Column(db.String(255), nullable=False)

    predicted_class = db.Column(db.String(100), nullable=False)

    confidence = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship(
        'User',
        backref=db.backref('predictions', lazy=True)
    )

# =========================
# LOGIN USER LOADER
# =========================

@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))

# =========================
# HOME PAGE
# =========================

@app.route('/')
def index():

    return render_template('index.html')

# =========================
# REGISTER
# =========================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if current_user.is_authenticated:

        return redirect(url_for('dashboard'))

    if request.method == 'POST':

        username = request.form.get('username')

        email = request.form.get('email')

        password = request.form.get('password')

        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:

            flash('Passwords do not match!', 'danger')

            return render_template('register.html')

        if User.query.filter_by(username=username).first():

            flash('Username already exists!', 'danger')

            return render_template('register.html')

        if User.query.filter_by(email=email).first():

            flash('Email already exists!', 'danger')

            return render_template('register.html')

        user = User(username=username, email=email)

        user.set_password(password)

        db.session.add(user)

        db.session.commit()

        flash('Registration successful! Please login.', 'success')

        return redirect(url_for('login'))

    return render_template('register.html')

# =========================
# LOGIN
# =========================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if current_user.is_authenticated:

        return redirect(url_for('dashboard'))

    if request.method == 'POST':

        username = request.form.get('username')

        password = request.form.get('password')

        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):

            login_user(user, remember=remember)

            next_page = request.args.get('next')

            flash('Login successful!', 'success')

            return redirect(next_page) if next_page else redirect(url_for('dashboard'))

        else:

            flash('Invalid username or password!', 'danger')

    return render_template('login.html')

# =========================
# LOGOUT
# =========================

@app.route('/logout')
@login_required
def logout():

    logout_user()

    flash('You have been logged out.', 'info')

    return redirect(url_for('index'))

# =========================
# DASHBOARD
# =========================

@app.route('/dashboard')
@login_required
def dashboard():

    user_predictions = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        PredictionHistory.created_at.desc()
    ).limit(10).all()

    return render_template(
        'upload.html',
        predictions=user_predictions
    )

# =========================
# FILE VALIDATION
# =========================

def allowed_file(filename):

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {
               'png',
               'jpg',
               'jpeg',
               'gif',
               'bmp'
           }

# =========================
# UPLOAD AND PREDICT
# =========================

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():

    if 'file' not in request.files:

        flash('No file selected!', 'danger')

        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '':

        flash('No file selected!', 'danger')

        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        try:

            # =========================
            # PREDICT PLANT
            # =========================

            result = predict_plant(filepath)

            plant_name = result["plant_name"]

            confidence = result["confidence"]

            about = result["about"]

            uses = result["uses"]

            benefits = result["benefits"]

            # =========================
            # SAVE TO DATABASE
            # =========================

            prediction_record = PredictionHistory(

                user_id=current_user.id,

                filename=filename,

                predicted_class=plant_name,

                confidence=confidence
            )

            db.session.add(prediction_record)

            db.session.commit()

            # =========================
            # IMAGE TO BASE64
            # =========================

            with open(filepath, 'rb') as img_file:

                img_data = base64.b64encode(
                    img_file.read()
                ).decode('utf-8')

            # =========================
            # SHOW RESULT
            # =========================

            return render_template(

                'result.html',

                prediction=plant_name,

                confidence=confidence,

                about=about,

                uses=uses,

                benefits=benefits,

                image_data=img_data,

                filename=filename
            )

        except Exception as e:

            flash(f'Error processing image: {str(e)}', 'danger')

            return redirect(url_for('dashboard'))

    else:

        flash('Invalid file type! Please upload an image.', 'danger')

        return redirect(url_for('dashboard'))

# =========================
# API PREDICT
# =========================

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():

    if 'file' not in request.files:

        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':

        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        try:

            result = predict_plant(filepath)

            return jsonify({

                'prediction': result["plant_name"],

                'confidence': result["confidence"],

                'about': result["about"],

                'uses': result["uses"],

                'benefits': result["benefits"],

                'filename': filename
            })

        except Exception as e:

            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

# =========================
# HISTORY
# =========================

@app.route('/history')
@login_required
def prediction_history():

    page = request.args.get('page', 1, type=int)

    per_page = 10

    predictions = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        PredictionHistory.created_at.desc()
    ).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )

    return render_template(
        'history.html',
        predictions=predictions
    )

# =========================
# INITIALIZE DATABASE
# =========================

with app.app_context():

    db.create_all()

# =========================
# RUN APP
# =========================

if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )