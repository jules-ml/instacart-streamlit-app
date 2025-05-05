# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations
from collections import Counter
import os
os.makedirs("data", exist_ok=True)

warnings.filterwarnings('ignore')

# Define base path
base_path = 'C:/Users/jules/OneDrive/Documents/UCF/Winter 2025/ML/Instacart/'

# Load Data
print("--- Loading Data ---")
# Load smaller files
aisles = pd.read_csv(f'{base_path}aisles.csv')
departments = pd.read_csv(f'{base_path}departments.csv')
products = pd.read_csv(f'{base_path}products.csv')

# Load order_products__prior.csv for prior order data
prior_orders_df = pd.read_csv(f'{base_path}order_products__prior.csv')

# Ensure product data is loaded into products_df for consistency
products_df = products.copy()

# --- Feature Engineering: Department and Aisle Reorder Rates ---

# Department reorder rate
dept_reorder = prior_orders_df.merge(products_df, on='product_id') \
    .groupby('department_id')['reordered'].mean().reset_index()
dept_reorder.columns = ['department_id', 'department_reorder_rate']

# Aisle reorder rate
aisle_reorder = prior_orders_df.merge(products_df, on='product_id') \
    .groupby('aisle_id')['reordered'].mean().reset_index()
aisle_reorder.columns = ['aisle_id', 'aisle_reorder_rate']


print("Smaller files loaded successfully.")
print(f"Aisles: {aisles.shape}")
print(f"Departments: {departments.shape}")
print(f"Products: {products.shape}")

# Load orders data
print("\nLoading orders data...")
orders = pd.read_csv(f'{base_path}orders.csv',
                     dtype={'order_id': 'int32',
                            'user_id': 'int32',
                            'order_number': 'int16',
                            'order_dow': 'int8',
                            'order_hour_of_day': 'int8',
                            'days_since_prior_order': 'float32'}) # Specify dtype
print(f"Orders shape: {orders.shape}")

# Load order_products_train
print("\nLoading order_products_train...")
order_products_train = pd.read_csv(f'{base_path}order_products__train.csv',
                                  dtype={'order_id': 'int32',
                                         'product_id': 'int32',
                                         'add_to_cart_order': 'int16',
                                         'reordered': 'int8'})
print(f"Order products train shape: {order_products_train.shape}")
# Note: We load train data but focus EDA on prior data for now.

# Load a sample of order_products_prior (due to size)
print("\nLoading a sample of order_products_prior...")
sample_rate = 0.05  # Use 5% of the data - adjust if needed/possible
try:
    order_products_prior = pd.read_csv(f'{base_path}order_products__prior.csv',
                                   skiprows=lambda i: i>0 and np.random.rand() > sample_rate,
                                   dtype={'order_id': 'int32',
                                          'product_id': 'int32',
                                          'add_to_cart_order': 'int16',
                                          'reordered': 'int8'})
    print(f"Order products prior sample shape: {order_products_prior.shape}")
    print(f"NOTE: Using a {sample_rate*100:.1f}% sample of prior orders for subsequent analysis.")
except FileNotFoundError:
    print(f"Error: Could not find order_products__prior.csv at {base_path}")
    print("Subsequent analysis using prior orders will fail. Please check the file path.")
    order_products_prior = pd.DataFrame() # Create empty df to avoid errors later


# Initial Data Checks & Merging
print("\n--- Initial Data Checks & Merging ---")

# Missing Value Analysis
print("\n--- Missing Value Analysis ---")
print("Checking for missing values:")
print("\nAisles Info:")
print(aisles.isnull().sum())
print("\nDepartments Info:")
print(departments.isnull().sum())
print("\nProducts Info:")
print(products.isnull().sum())

print("\nOrders Info:")
print(orders.isnull().sum())
# Analyze the significant NaNs in 'days_since_prior_order'
nan_days_prior = orders['days_since_prior_order'].isnull().sum()
first_orders = orders[orders['order_number'] == 1].shape[0]
print(f"\nMissing values in 'days_since_prior_order': {nan_days_prior}")
print(f"Number of first orders (order_number == 1): {first_orders}")
print(f"Match confirms NaNs correspond to first orders.")

# For analysis visualization, we can fill NaNs. Choice depends on goal.
# Fill with 0 can be misleading since 0 is a valid value
#  Fill with a distinct value (e.g., -1) to separate first orders
orders['days_since_prior_order_filled'] = orders['days_since_prior_order'].fillna(-1)
# Keep the original NaNs for potential feature engineering later if needed.

print("\nOrder Products Train Info:")
print(order_products_train.isnull().sum())
if not order_products_prior.empty:
    print("\nOrder Products Prior (Sample) Info:")
    print(order_products_prior.isnull().sum())
else:
    print("\nOrder Products Prior (Sample) Info: Skipped due to loading error.")


# Merge products with aisles and departments
print("\nMerging products with categories...")
products_info = products.merge(aisles, on='aisle_id', how='left')
products_info = products_info.merge(departments, on='department_id', how='left')
print(f"Products with categories shape: {products_info.shape}")
# print(products_info.head()) # Keep if needed

# Basic EDA & Visualizations
print("\n--- Basic EDA & Visualizations ---")

# distribution of products across departments
dept_counts = products_info['department'].value_counts()
plt.figure(figsize=(10, 6))
dept_counts.plot(kind='bar')
plt.title('Number of Products by Department')
plt.ylabel('Number of Products')
plt.xlabel('Department')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Basic order stats
print("\n--- Order Statistics ---")
total_orders = orders['order_id'].nunique()
total_customers = orders['user_id'].nunique()
print(f"Total orders: {total_orders}")
print(f"Total customers: {total_customers}")
# Avg orders per customer - better calculated in customer features section

# Order Pattern Analysis
print("\n--- Order Pattern Analysis ---")

# When do people shop? (Day of week & hour of day)
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
# Use integer indices directly for plotting, map labels for clarity
day_counts = orders['order_dow'].value_counts().sort_index()
day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] # Abbreviated for space
sns.barplot(x=day_counts.index, y=day_counts.values, palette="viridis")
plt.title('Orders by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Orders')
plt.xticks(ticks=range(len(day_names)), labels=day_names, rotation=45)

plt.subplot(1, 2, 2)
hour_counts = orders['order_hour_of_day'].value_counts().sort_index()
sns.barplot(x=hour_counts.index, y=hour_counts.values, palette="viridis")
plt.title('Orders by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()

# Analyze time between orders
plt.figure(figsize=(12, 6))
# Use the filled data for visualization, excluding the -1 marker for first orders
sns.histplot(orders[orders['days_since_prior_order_filled'] != -1]['days_since_prior_order'], bins=30, kde=False)
plt.title('Distribution of Days Since Prior Order (Excluding First Orders)')
plt.xlabel('Days Since Prior Order')
plt.ylabel('Frequency')
plt.axvline(x=orders['days_since_prior_order'].mean(), color='r', linestyle='--',
            label=f"Mean: {orders['days_since_prior_order'].mean():.2f} days")
plt.legend()
plt.show()

# Relationship between order number and days since prior order
order_timing_agg = orders[orders['order_number'] > 1].groupby('order_number')['days_since_prior_order'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=order_timing_agg, x='order_number', y='days_since_prior_order')
plt.title('Average Days Since Prior Order vs. Order Number')
plt.xlabel('Customer Order Number')
plt.ylabel('Average Days Since Prior Order')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Does the time between orders stabilize?

# Build Customer Transactional Dataset
# This section creates a DataFrame where each row is a unique user
# and columns represent aggregated features of their purchasing behavior.
# This is crucial for models predicting behavior based on user history.

print("\n--- 5. Building Enhanced Customer Transactional Dataset ---")

# Start with basic aggregations from the 'orders' table
print("Calculating basic order stats per user...")
customer_transactional_data = orders.groupby('user_id').agg(
    total_orders=('order_number', 'max'),
    # Calculate avg/median days between orders (ignores first order NaNs implicitly)
    avg_days_between_orders=('days_since_prior_order', 'mean'),
    median_days_between_orders=('days_since_prior_order', 'median'),
    # Get the DOW/Hour of their first and last recorded order
    # first_order_day=('order_dow', 'first'), # Less commonly used
    # last_order_day=('order_dow', 'last'),   # Less commonly used
    avg_order_dow=('order_dow', 'mean'), # Avg DOW might indicate preference
    avg_order_hour=('order_hour_of_day', 'mean'), # Avg hour might indicate preference
    # Find the most frequent day and hour
    # .mode()[0] selects the first mode if multiple exist
    most_frequent_dow=('order_dow', lambda x: x.mode()[0] if not x.mode().empty else -1),
    most_frequent_hour=('order_hour_of_day', lambda x: x.mode()[0] if not x.mode().empty else -1)
).reset_index()

print(f"Initial customer data shape: {customer_transactional_data.shape}")

# Add features requiring product information (using the sampled prior data)
# These calculations depend on 'order_products_prior' being loaded successfully
if not order_products_prior.empty:
    print("\nMerging with sampled prior product data to add more features...")
    # Merge orders with the sampled prior products - use 'inner' to keep only orders present in the sample
    # Note: This means users with ONLY 'train' orders won't appear here
    orders_with_prior_products = orders.merge(order_products_prior, on='order_id', how='inner')

    # Calculate total items purchased per user (in sample)
    print("Calculating total items and unique products per user...")
    user_item_counts = orders_with_prior_products.groupby('user_id').agg(
        total_items_purchased_prior=('product_id', 'size'), # 'size' counts rows
        total_unique_products_prior=('product_id', 'nunique')
    ).reset_index()

    # Calculate average basket size per user (in sample)
    # First get basket size per order
    basket_sizes_sampled = orders_with_prior_products.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
    # Then average per user
    avg_basket_size_sampled = basket_sizes_sampled.groupby('user_id')['basket_size'].mean().reset_index()
    avg_basket_size_sampled.rename(columns={'basket_size': 'avg_basket_size_prior'}, inplace=True)

    # Calculate overall reorder rate per user (in sample)
    print("Calculating reorder rate per user...")
    user_reorder_rate = orders_with_prior_products.groupby('user_id')['reordered'].mean().reset_index()
    user_reorder_rate.rename(columns={'reordered': 'user_reorder_rate_prior'}, inplace=True)

    # Feature: Favorite Department/Aisle 
    # This requires merging with products_info
    if not products_info.empty:
        print("Calculating favorite department per user (can be slow)...")
        # Merge product details into the orders_with_products
        orders_prods_details = orders_with_prior_products.merge(products_info, on='product_id', how='left')

        # Count purchases per department per user
        user_dept_counts = orders_prods_details.groupby(['user_id', 'department'])['order_id'].count().reset_index(name='dept_purchase_count')

        if not user_dept_counts.empty:
            # Find the department with the max count for each user
            # idx = user_dept_counts.groupby(['user_id'])['dept_purchase_count'].transform(max) == user_dept_counts['dept_purchase_count'] # Handles ties by keeping all max
            # Using idxmax is simpler but only keeps the first max department in case of a tie
            idx = user_dept_counts.groupby(['user_id'])['dept_purchase_count'].idxmax()
            fav_dept = user_dept_counts.loc[idx, ['user_id', 'department']].rename(columns={'department': 'favorite_department_prior'})

            # Merge favorite department back
            customer_transactional_data = customer_transactional_data.merge(fav_dept, on='user_id', how='left')
        else:
             print("Skipping favorite department calculation (no data after merge).")
             customer_transactional_data['favorite_department_prior'] = np.nan # Add empty column if calc failed

    else:
        print("Skipping favorite department calculation (products_info is empty).")
        customer_transactional_data['favorite_department_prior'] = np.nan # Add empty column

    # Merge all new features back into the main customer transactional dataframe
    print("Merging all calculated user features...")
    customer_transactional_data = customer_transactional_data.merge(user_item_counts, on='user_id', how='left')
    customer_transactional_data = customer_transactional_data.merge(avg_basket_size_sampled, on='user_id', how='left')
    customer_transactional_data = customer_transactional_data.merge(user_reorder_rate, on='user_id', how='left')

    # Fill NaNs that might result from left merges (e.g., users with no prior data in sample)
    # Fill NaNs for numerical aggregates maybe with 0 or based on context
    cols_to_fill_zero = ['total_items_purchased_prior', 'total_unique_products_prior', 'avg_basket_size_prior', 'user_reorder_rate_prior']
    for col in cols_to_fill_zero:
        if col in customer_transactional_data.columns:
             customer_transactional_data[col].fillna(0, inplace=True)
    # For favorite department, NaN might be acceptable or fill with 'Unknown'
    if 'favorite_department_prior' in customer_transactional_data.columns:
        customer_transactional_data['favorite_department_prior'].fillna('Unknown', inplace=True)


else:
    print("\nSkipping calculation of product-related customer features due to prior orders loading error or empty sample.")
    # Add empty columns if they weren't created
    if 'total_items_purchased_prior' not in customer_transactional_data.columns: customer_transactional_data['total_items_purchased_prior'] = 0
    if 'total_unique_products_prior' not in customer_transactional_data.columns: customer_transactional_data['total_unique_products_prior'] = 0
    if 'avg_basket_size_prior' not in customer_transactional_data.columns: customer_transactional_data['avg_basket_size_prior'] = 0
    if 'user_reorder_rate_prior' not in customer_transactional_data.columns: customer_transactional_data['user_reorder_rate_prior'] = 0
    if 'favorite_department_prior' not in customer_transactional_data.columns: customer_transactional_data['favorite_department_prior'] = 'Unknown'


print(f"\nFinal Customer Transactional Dataset shape: {customer_transactional_data.shape}")
print("Sample of the final Customer Transactional Dataset:")
print(customer_transactional_data.head())

# Product Popularity & Reorder Analysis (Using Sampled Prior Data)
print("\n--- Product Popularity & Reorder Analysis (Based on Sampled Prior Data) ---")
if not order_products_prior.empty and not products_info.empty:
    # Merge order_products_prior (sample) with product info
    popular_products_sampled = order_products_prior.merge(products_info, on='product_id')

    # Top 15 products (from sample)
    top_products_sampled = popular_products_sampled['product_name'].value_counts().head(15)
    plt.figure(figsize=(12, 6))
    top_products_sampled.plot(kind='bar')
    plt.title('Top 15 Most Popular Products (from Sample)')
    plt.ylabel('Number of Orders (in Sample)')
    plt.xlabel('Product Name')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Department popularity (from sample)
    dept_popularity_sampled = popular_products_sampled.groupby('department')['order_id'].count().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    dept_popularity_sampled.plot(kind='bar')
    plt.title('Popularity of Departments (from Sample)')
    plt.ylabel('Number of Orders (in Sample)')
    plt.xlabel('Department')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Reorder Analysis
    print("\n--- Reorder Analysis (Based on Sampled Prior Data) ---")

    # Calculate reorder ratio by product (from sample)
    product_reorders_sampled = order_products_prior.groupby('product_id').agg(
        reorder_count=('reordered', 'sum'),
        purchase_count=('reordered', 'count') # 'count' includes both reordered=1 and reordered=0
    )
    # product_reorders_sampled.columns = ['reorder_count', 'purchase_count'] # Alternative naming
    product_reorders_sampled['reorder_ratio'] = product_reorders_sampled['reorder_count'] / product_reorders_sampled['purchase_count']

    # Filter to products with at least 10 purchases in the sample
    min_purchases_threshold = 10
    frequent_products_sampled = product_reorders_sampled[product_reorders_sampled['purchase_count'] > min_purchases_threshold].copy()

    # Get product names for top reordered products
    top_reordered_sampled = frequent_products_sampled.sort_values('reorder_ratio', ascending=False).head(15)
    top_reordered_with_names_sampled = top_reordered_sampled.merge(products_info, left_index=True, right_on='product_id')

    print(f"\nTop 15 Most Frequently Reordered Products (Min {min_purchases_threshold} purchases in Sample):")
    print(top_reordered_with_names_sampled[['product_name', 'department', 'reorder_ratio', 'purchase_count']])

else:
    print("\nSkipping Product Popularity & Reorder Analysis due to data loading issues.")

# Basket Analysis (Using Sampled Prior Data) 
print("\n--- Basket Analysis (Based on Sampled Prior Data) ---")
if not order_products_prior.empty:
    # Basket size distribution (from sample)
    basket_sizes = order_products_prior.groupby('order_id')['product_id'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(basket_sizes, bins=max(1, min(basket_sizes.max(), 50)), kde=True) # Adjust bins
    plt.title('Distribution of Basket Sizes (from Sample)')
    plt.xlabel('Number of Products per Basket')
    plt.ylabel('Frequency (in Sample)')
    mean_basket_size = basket_sizes.mean()
    plt.axvline(x=mean_basket_size, color='r', linestyle='--',
                label=f'Mean: {mean_basket_size:.2f} products')
    plt.legend()
    plt.show()

    # --- Product Association Analysis (from Sample) ---
    print("\n--- Product Association Analysis (Based on Sampled Prior Data) ---")
    # This part is computationally intensive, especially finding pairs.
    # Using the existing sampled 'order_products_prior'
    # Filter orders with at least 2 items to find pairs
    order_counts = order_products_prior['order_id'].value_counts()
    multi_item_orders = order_counts[order_counts >= 2].index
    order_pairs_df = order_products_prior[order_products_prior['order_id'].isin(multi_item_orders)].copy()

    # Group by order_id and get product lists
    order_products_grouped = order_pairs_df.groupby('order_id')['product_id'].apply(list)

    # Find pairs (can still be slow - maybe sample the orders further if needed)
    # Consider sampling 'order_products_grouped' if it's too large
    if len(order_products_grouped) > 50000: # Arbitrary limit, adjust based on memory
        print(f"Sampling {50000} orders for pair analysis due to large size...")
        order_products_grouped = order_products_grouped.sample(50000, random_state=42)

    product_pairs = []
    for product_list in order_products_grouped:
        # Sort the list to ensure ('A', 'B') is the same as ('B', 'A')
        sorted_list = sorted(product_list)
        pairs = list(combinations(sorted_list, 2))
        product_pairs.extend(pairs)

    # Count occurrences of each pair
    pair_counts = Counter(product_pairs)

    # Get the most common pairs
    top_pairs = pair_counts.most_common(15) # Increased to top 15
    print("\nTop 15 Product Pairs Purchased Together (from Sample):")
    if not products.empty:
        for pair, count in top_pairs:
            try:
                prod1_name = products.loc[products['product_id'] == pair[0], 'product_name'].iloc[0]
                prod2_name = products.loc[products['product_id'] == pair[1], 'product_name'].iloc[0]
                print(f"{prod1_name} + {prod2_name}: {count} times")
            except IndexError:
                print(f"Product ID {pair[0]} or {pair[1]} not found in products table.")
    else:
         print("Cannot display product names as products DataFrame is empty.")

else:
    print("\nSkipping Basket & Association Analysis due to prior orders loading error.")


print("\n--- EDA Phase Complete ---")

# Department reorder rate
dept_reorder = prior_orders_df.merge(products_df, on='product_id') \
    .groupby('department_id')['reordered'].mean().reset_index()
dept_reorder.columns = ['department_id', 'department_reorder_rate']

# Aisle reorder rate
aisle_reorder = prior_orders_df.merge(products_df, on='product_id') \
    .groupby('aisle_id')['reordered'].mean().reset_index()
aisle_reorder.columns = ['aisle_id', 'aisle_reorder_rate']

product_reorder_freq = prior_orders_df.groupby('product_id')['reordered'].agg(['mean', 'sum']).reset_index()
product_reorder_freq.columns = ['product_id', 'product_reorder_prob', 'product_total_reorders']

# Merge user_id into prior_orders_df
prior_orders_df = prior_orders_df.merge(
    orders[['order_id', 'user_id', 'order_number']],
    how='left',
    on='order_id'
)

user_prod_stats = prior_orders_df.groupby(['user_id', 'product_id']).agg({
    'reordered': 'sum',
    'order_number': 'max'
}).reset_index()

user_prod_stats.columns = ['user_id', 'product_id', 'user_product_reorder_count', 'user_product_last_order_number']

# --- USER RECENCY FEATURES ---
print("🔁 Generating user-level recency features...")

# Filter for prior + train orders only (exclude test)
user_orders = orders[orders['eval_set'].isin(['prior', 'train'])]

# Avg & median days since order
user_recency = user_orders.groupby('user_id')['days_since_prior_order'].agg(
    user_avg_days_since_order='mean',
    user_median_days_since_order='median'
).reset_index()

# Most recent order's recency (last known)
last_orders = user_orders.sort_values(['user_id', 'order_number'], ascending=[True, False]) \
    .groupby('user_id').first().reset_index()

user_recency = user_recency.merge(
    last_orders[['user_id', 'days_since_prior_order']],
    on='user_id',
    how='left'
).rename(columns={'days_since_prior_order': 'user_last_order_days_ago'})

user_recency.head()

# Export department/aisle reorder rates
dept_reorder.to_csv("department_reorder.csv", index=False)
aisle_reorder.to_csv("aisle_reorder.csv", index=False)

# Export user-product stats (if not done already)
user_prod_stats.to_csv("user_product_history.csv", index=False)

# --- BASIC USER FEATURES ---
user_stats = orders[orders['eval_set'].isin(['prior', 'train'])] \
    .groupby('user_id').agg(
        user_total_orders=('order_number', 'max'),
        user_avg_basket_size=('order_id', 'count')
    ).reset_index()

# --- USER RECENCY FEATURES ---
user_orders = orders[orders['eval_set'].isin(['prior', 'train'])]

user_recency = user_orders.groupby('user_id')['days_since_prior_order'].agg(
    user_avg_days_since_order='mean',
    user_median_days_since_order='median'
).reset_index()

last_orders = user_orders.sort_values(['user_id', 'order_number'], ascending=[True, False]) \
    .groupby('user_id').first().reset_index()

user_recency = user_recency.merge(
    last_orders[['user_id', 'days_since_prior_order']],
    on='user_id',
    how='left'
).rename(columns={'days_since_prior_order': 'user_last_order_days_ago'})

# --- COMBINE + EXPORT ---
user_features = user_stats.merge(user_recency, on='user_id', how='left')

# Create 'data' directory before saving
import os
os.makedirs("data", exist_ok=True)

# save
user_features.to_csv("data/user_features.csv", index=False)

# Assuming prior_orders_df is your cleaned order_products__prior + orders merge
user_prod_stats = prior_orders_df.groupby(['user_id', 'product_id']).agg({
    'reordered': 'sum',
    'order_number': 'max'
}).reset_index()

user_prod_stats.columns = [
    'user_id', 'product_id',
    'user_product_reorder_count',
    'user_product_last_order_number'
]

user_prod_stats.to_csv("data/user_product_stats.csv", index=False)

product_stats = prior_orders_df.groupby('product_id')['reordered'].agg([
    ('product_total_orders', 'count'),
    ('product_reorders', 'sum')
]).reset_index()

product_stats['product_reorder_prob'] = (
    product_stats['product_reorders'] / product_stats['product_total_orders']
)

product_stats.to_csv("data/product_stats.csv", index=False)

# --- BASIC USER FEATURES ---
user_stats = orders[orders['eval_set'].isin(['prior', 'train'])] \
    .groupby('user_id').agg(
        user_total_orders=('order_number', 'max'),
        user_avg_basket_size=('order_id', 'count')
    ).reset_index()

# Combine stats and recency into one user_features DataFrame
user_features = user_stats.merge(user_recency, on='user_id', how='left')

# ✅ Create 'data' folder if it doesn’t exist
os.makedirs("data", exist_ok=True)

# ✅ Save the final user features CSV
user_features.to_csv("data/user_features.csv", index=False)

print("✅ user_features.csv saved to /data")
