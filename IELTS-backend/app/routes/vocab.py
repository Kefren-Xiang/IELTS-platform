from flask import Blueprint, jsonify
from app.utils.database import query_db

vocab_bp = Blueprint('vocab', __name__, url_prefix='/api')

@vocab_bp.route('/vocab', methods=['GET'])
def get_vocab():
    data = query_db("SELECT * FROM vocab")
    return jsonify(data)
