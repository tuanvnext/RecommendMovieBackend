from flask import Flask
from flask import jsonify
from movielen import recommend
import pandas as pd


app = Flask(__name__)
recommend_item = None
@app.route('/')
def home():
    return 'Helloworld'

@app.route('/users', methods=['GET'])
def get_all_user():
    abs_user = recommend.User()
    list_user = abs_user.get_all_user()
    list_user = list_user.to_json(orient='records')
    return jsonify(list_user)

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user_by_id(user_id):
    abs_user = recommend.User()
    user = abs_user.get_user_by_id(user_id)
    user = user.to_json(orient='records')
    return user

@app.route('/users/<int:user_id>/items', methods=['GET'])
def get_top_items(user_id):
    id_items = []
    if recommend_item is not None:
        id_items = recommend_item.rcm(user_id)
    all_items = recommend.Item().get_all_item()
    items_recommend = all_items.iloc[id_items, :]
    print(type(items_recommend))
    items_recommend = items_recommend.to_json(orient='records')
    return items_recommend

if __name__ == '__main__':
    recommend_item = recommend.process()
    app.run(debug=True)