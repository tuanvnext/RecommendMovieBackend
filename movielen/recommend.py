import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class CollaborativeFiltering(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity):
        self.Y_data = Y_data
        self.k = k # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def normalize_Y(self):
        users = self.Y_data[:, 0]  # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important
        # for both memory and computing efficiency. For example, if #user = 1M,
        # #item = 100k, then shape of the rating matrix would be (100k, 1M),
        # you may not have enough memory to store this. Then, instead, we store
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        return self.__pred(u, i, normalized)

    def recommend(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)

        return recommended_items

    def print_recommendation(self):
        """
        print all items which should be recommended for each user
        """
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            print('    Recommend item(s):', recommended_items, 'for user', u)
    def rcm(self, u):
        recommended_items = self.recommend(u)
        return recommended_items

class Item(object):
    def __init__(self, path='/home/tuanlv/PycharmProjects/RecommendionSystemNetflix/movielen/data/ml-100k/u.item'):
        self.path = path
        self.r_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
        self.__load_item()

    def __load_item(self):
        self.items = pd.read_csv('/home/tuanlv/PycharmProjects/RecommendionSystemNetflix/movielen/data/ml-100k/u.item', sep='|', names=self.r_cols, encoding='latin-1')

    def get_cols_name(self):
        return self.r_cols

    def get_item_by_id(self, id):
        item = self.items[id, :]
        return item

    def get_all_item(self):
        return self.items

class User(object):
    def __init__(self, path='/home/tuanlv/PycharmProjects/RecommendionSystemNetflix/movielen/data/ml-100k/u.user'):
        self.path = path
        self.r_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        self.__load_user()

    def __load_user(self):
        self.users = pd.read_csv(self.path, sep='|', names=self.r_cols, encoding='latin-1')

    def get_cols_name(self):
        return self.r_cols

    def get_user_by_id(self, id):
        user = self.users.iloc[id, :]
        return user

    def get_all_user(self):
        return self.users

def process():
    r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('/home/tuanlv/PycharmProjects/RecommendionSystemNetflix/movielen/data/ml-100k/u.data',
                          sep='\t', names=r_cols, encoding='latin-1')
    Y_data = ratings.as_matrix()
    rs = CollaborativeFiltering(Y_data, k=2)
    rs.fit()
    return rs

if __name__ == '__main__':
    rs = process()
    rs.rcm(50)
    # user = User()
    # all_users = user.get_all_user()
    # json_user = all_users.to_json(orient='records')
    # print(json_user)