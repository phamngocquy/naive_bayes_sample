import os

import pandas as pd

import mysql.connector

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}


def loadData():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(buffered=True)

    product_df = pd.read_sql_query("select products.name,categories.name as category from products "
                                   "inner join categories on products.category_id = categories.id "
                                   "where products.category_id != 9999 and products.is_active = 1", cnx)
    path = os.path.join("",
                        'products.csv')
    product_df.to_csv(path)
    cnx.close()


if __name__ == '__main__': loadData()
