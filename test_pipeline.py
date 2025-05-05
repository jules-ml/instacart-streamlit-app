from utils.pipeline import predict_user_top_bottom_products

# Try a few known user IDs
test_users = [100, 200, 300, 400, 500]

for user_id in test_users:
    print(f"\n🧪 Testing user_id: {user_id}")
    top5, bottom5 = predict_user_top_bottom_products(user_id)
    
    print("🥇 Top 5 Predictions:")
    for product in top5:
        print(f"   - {product}")
        
    print("🚫 Bottom 5 Predictions:")
    for product in bottom5:
        print(f"   - {product}")
