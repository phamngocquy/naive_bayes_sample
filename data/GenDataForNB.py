import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector
from unidecode import unidecode

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}


def load_data():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(buffered=True)
    cursor.execute("select products.name,categories.name as category,category_id from products "
                   "inner join categories on products.category_id = categories.id "
                   "where products.category_id != 9999 AND products.is_active = True")
    data = cursor.fetchall()
    name_product_list_df = []
    for row in data:
        d = {"name": row[0], "category": row[1], "category_id": row[2]}
        name_product_list_df.append(d)
        d = {"name": unidecode(row[0]), "category": row[1], "category_id": row[2]}
        name_product_list_df.append(d)

    product_df = pd.DataFrame(name_product_list_df)
    # product_df = pd.read_sql_query("select products.name,categories.name as category from products "
    #                                "inner join categories on products.category_id = categories.id "
    #                                "where products.category_id != 9999 and products.is_active = 1", cnx)

    product_df["category_id"].value_counts().plot(kind='bar')
    plt.show()
    path = os.path.join("",
                        'products.csv')
    product_df.to_csv(path)
    cnx.close()


if __name__ == '__main__':
    load_data()
