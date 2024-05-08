import os
import pandas as pd


current_directory = os.path.dirname(__file__)

# Construct the path to the CSV file
csv_file_path = os.path.join(current_directory, "../../data/sales_clothes.csv")

# Load the dataset
data = pd.read_csv(csv_file_path)
data.dropna(inplace=True)


data = pd.get_dummies(data)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['price', 'retail_price', 'units_sold', 'rating', 'rating_count', 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count', 'badges_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'product_variation_inventory', 'shipping_option_price', 'countries_shipped_to', 'inventory_total', 'merchant_rating']] = scaler.fit_transform(data[['price', 'retail_price', 'units_sold', 'rating', 'rating_count', 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count', 'badges_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'product_variation_inventory', 'shipping_option_price', 'countries_shipped_to', 'inventory_total', 'merchant_rating']])
